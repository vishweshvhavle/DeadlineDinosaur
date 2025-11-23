# Copyright (c) 2025 Harbin Institute of Technology, Huawei Noah's Ark Lab
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import math
import torch
from typing import List

class TrainingScheduler():
	"""
	Hybrid training scheduler: DashGaussian momentum → LiteGS fixed target
	
	Strategy:
	- Phase 1 (scale > 1): Momentum-based adaptive growth (DashGaussian)
	- Phase 2 (scale = 1): Fixed linear target (LiteGS)
	"""
	def __init__(self, opt, dens, pipe, init_n_gaussian: int, original_images: List) -> None:

		self.max_steps = opt.iterations
		self.init_n_gaussian = init_n_gaussian

		self.densify_mode = pipe.densify_mode
		self.densify_until_iter = dens.densify_until
		self.densification_interval = dens.densification_interval
		
		self.resolution_mode = pipe.resolution_mode

		self.start_significance_factor = 4
		self.max_reso_scale = 8
		self.reso_sample_num = 32
		self.max_densify_rate_per_step = 0.2
		self.reso_scales = None
		self.reso_level_significance = None
		self.reso_level_begin = None
		self.increase_reso_until = dens.reso_until_iter
		self.next_i = 2

		# === HYBRID BUDGETING PARAMETERS ===
		self.budgeting_mode = "hybrid"  # "momentum" or "fixed" or "hybrid"
		
		# Phase 1: Momentum-based (DashGaussian)
		self.momentum = 5 * init_n_gaussian  # Start with 5x initial
		self.max_n_gaussian = self.init_n_gaussian + self.momentum
		self.integrate_factor = 0.98  # γ in Eq.5
		self.eta = 1.0  # η in Eq.5
		self.momentum_step_cap = 1000000  # Cap per-step addition
		
		# Phase 2: Fixed target (LiteGS)
		self.fixed_target = None  # Will be set when transitioning
		self.transition_iteration = None  # When we hit scale=1
		self.has_transitioned = False
		
		# If user explicitly sets target, use fixed mode throughout
		if pipe.max_n_gaussian > 0:
			self.budgeting_mode = "fixed"
			self.fixed_target = pipe.max_n_gaussian
			self.momentum = -1
			print(f"[SCHEDULER] Fixed mode: target={self.fixed_target} primitives")
		else:
			print(f"[SCHEDULER] Hybrid mode: momentum → fixed at full resolution")
			print(f"[SCHEDULER] Initial Pfin={self.max_n_gaussian} (5x init)")
		
		print(f"[SCHEDULER] Densify until iteration: {self.densify_until_iter}")
		print(f"[SCHEDULER] Resolution mode: {self.resolution_mode}")
		
		# Generate schedulers
		self.init_reso_scheduler(original_images)

	def update_momentum(self, momentum_step):
		"""
		Update momentum using DashGaussian Eq.5: Pfin = max(Pfin, γ·Pfin + η·Padd)
		
		Only applies during Phase 1 (progressive resolution).
		"""
		# Skip if in fixed mode or already transitioned
		if self.momentum == -1 or self.has_transitioned:
			return
		
		if momentum_step is None or momentum_step == 0:
			return
		
		# DashGaussian Eq.5 with capping
		capped_add = min(self.momentum_step_cap, momentum_step)
		new_momentum = self.integrate_factor * self.momentum + self.eta * capped_add
		self.momentum = max(self.momentum, int(new_momentum))
		self.max_n_gaussian = self.init_n_gaussian + self.momentum
		
		print(f"[MOMENTUM] Phase 1: Pfin={self.max_n_gaussian:.0f} "
		      f"(momentum={self.momentum:.0f}, added={momentum_step})")

	def transition_to_fixed_target(self, current_iteration, current_n_primitives):
		"""
		Transition from momentum-based to fixed-target mode at full resolution.
		
		Captures current primitive count as the target and switches to linear growth.
		"""
		if self.has_transitioned:
			return
		
		self.has_transitioned = True
		self.transition_iteration = current_iteration
		
		# Use current momentum estimate as fixed target
		self.fixed_target = self.max_n_gaussian
		
		print(f"\n{'='*60}")
		print(f"[TRANSITION] Reached full resolution (scale=1)")
		print(f"[TRANSITION] Switching from momentum → fixed target")
		print(f"[TRANSITION] Captured target: {self.fixed_target:.0f} primitives")
		print(f"[TRANSITION] Current count: {current_n_primitives}")
		print(f"[TRANSITION] Remaining iterations: {self.densify_until_iter - current_iteration}")
		print(f"{'='*60}\n")

	def get_res_scale(self, iteration):
		"""Get current resolution scale."""
		if self.resolution_mode == "const":
			return 1
		elif self.resolution_mode == "freq":
			if iteration >= self.increase_reso_until:
				return 1
			if iteration < self.reso_level_begin[1]:
				return self.reso_scales[0]
			while iteration >= self.reso_level_begin[self.next_i]:
				self.next_i += 1
			i = self.next_i - 1
			i_now, i_nxt = self.reso_level_begin[i: i + 2]
			s_lst, s_now = self.reso_scales[i - 1: i + 1]
			scale = (1 / ((iteration - i_now) / (i_nxt - i_now) * (1/s_now**2 - 1/s_lst**2) + 1/s_lst**2))**0.5
			return max(1, int(scale))
		else:
			raise NotImplementedError(f"Resolution mode '{self.resolution_mode}' not implemented")
	
	def get_densify_rate(self, iteration, cur_n_gaussian, cur_scale=None):
		"""
		Compute densification rate using hybrid strategy:
		- Phase 1 (scale > 1): DashGaussian momentum-based
		- Phase 2 (scale = 1): LiteGS fixed linear target
		"""
		print(f"[DENSIFY] Getting densify rate at iteration {iteration}, Densify Mode: {self.densify_mode}")
		if self.densify_mode == "free":
			return 1.0
		
		elif self.densify_mode == "freq":
			assert cur_scale is not None, "cur_scale required for freq mode"

			print(f"[DENSIFY][Iter {iteration}] Current Scale: {cur_scale}, Current Primitives: {cur_n_gaussian}, Has Transitioned: {self.has_transitioned}")
			
			# Check if we've reached full resolution
			if cur_scale == 1 and not self.has_transitioned:
				self.transition_to_fixed_target(iteration, cur_n_gaussian)
			
			# === PHASE 2: Fixed Linear Target (LiteGS style) ===
			if self.has_transitioned:
				return None
			
			# === PHASE 1: Momentum-Based (DashGaussian style) ===
			else:
				# DashGaussian Eq.4 with power factor decay
				progress = iteration / self.densify_until_iter
				power_factor = 2.0 - progress  # 2.0 → 1.0
				
				denominator = cur_scale ** power_factor
				target_n_primitives = self.init_n_gaussian + \
				                     (self.max_n_gaussian - self.init_n_gaussian) / denominator
				
				# Smooth growth over remaining steps
				remaining_iters = self.increase_reso_until - iteration
				if remaining_iters <= 0:
					return 0.0
				
				remaining_steps = max(1, remaining_iters // self.densification_interval)
				total_to_add = max(0, target_n_primitives - cur_n_gaussian)
				per_step_add = total_to_add / remaining_steps
				densify_rate = per_step_add / self.init_n_gaussian
				
				if iteration % 100 == 0:
					print(f"[DENSIFY] Phase 1 (Momentum): iter={iteration}, scale={cur_scale}, "
					      f"power={power_factor:.2f}, target={target_n_primitives:.0f}, "
					      f"current={cur_n_gaussian}, rate={densify_rate:.3f}")
			
			# Clamp to prevent explosive growth
			densify_rate = max(0, min(densify_rate, self.max_densify_rate_per_step))
			return densify_rate
		
		else:
			raise NotImplementedError(f"Densify mode '{self.densify_mode}' not implemented")
	
	def lr_decay_from_iter(self):
		"""Determine when to start LR decay."""
		if self.resolution_mode == "const":
			return 1
		
		# Start decay when scale drops below 2
		for i, s in zip(self.reso_level_begin, self.reso_scales):
			if s < 2:
				return i
		
		return self.increase_reso_until

	def init_reso_scheduler(self, original_images):
		"""Initialize frequency-based resolution scheduler."""
		if self.resolution_mode != "freq":
			print(f"[ INFO ] Skipped resolution scheduler, mode is {self.resolution_mode}")
			return

		def compute_win_significance(significance_map: torch.Tensor, scale: float):
			if significance_map.dim() < 3:
				significance_map = significance_map.unsqueeze(0)
			
			h, w = significance_map.shape[-2:]
			c = ((h + 1) // 2, (w + 1) // 2)
			win_size = (max(1, int(h / scale)), max(1, int(w / scale)))
			
			h_start = max(0, c[0] - win_size[0] // 2)
			h_end = min(h, c[0] + (win_size[0] + 1) // 2)
			w_start = max(0, c[1] - win_size[1] // 2)
			w_end = min(w, c[1] + (win_size[1] + 1) // 2)

			win_significance = significance_map[..., h_start:h_end, w_start:w_end].sum().item()
			return win_significance
		
		def scale_solver(significance_map: torch.Tensor, target_significance: float):
			L, R, T = 0., 1., 64
			for _ in range(T):
				mid = (L + R) / 2
				if mid == 0:
					break
				win_significance = compute_win_significance(significance_map, 1 / mid)
				if win_significance < target_significance:
					L = mid
				else:
					R = mid
			return 1 / max(L, 1e-9)
		
		print("[ INFO ] Initializing resolution scheduler...")

		self.max_reso_scale = 8
		self.next_i = 2
		scene_freq_image = None
		
		for img_tensor in original_images:
			if img_tensor.dim() == 4:
				img_tensor = img_tensor.squeeze(0)
			if img_tensor.dim() != 3:
				raise ValueError(f"Image tensor wrong dimensions: {img_tensor.shape}")
			
			img_tensor = img_tensor.float()
			
			if img_tensor.shape[0] > 1:
				img = img_tensor.mean(dim=0)
			else:
				img = img_tensor.squeeze(0)

			img_fft_centered = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
			img_fft_centered_mod = (img_fft_centered.real.square() + img_fft_centered.imag.square()).sqrt()
			
			if scene_freq_image is None:
				scene_freq_image = img_fft_centered_mod
			else:
				scene_freq_image += img_fft_centered_mod

			e_total = img_fft_centered_mod.sum().item()
			e_min = e_total / self.start_significance_factor
			self.max_reso_scale = min(self.max_reso_scale, scale_solver(img_fft_centered_mod, e_min))

		modulation_func = math.log

		self.reso_scales = []
		self.reso_level_significance = []
		self.reso_level_begin = []
		scene_freq_image /= len(original_images)
		E_total = scene_freq_image.sum().item()
		E_min = compute_win_significance(scene_freq_image, self.max_reso_scale)
		
		E_min = max(E_min, 1e-9)
		E_total = max(E_total, E_min + 1e-9)
		
		self.reso_level_significance.append(E_min)
		self.reso_scales.append(self.max_reso_scale)
		self.reso_level_begin.append(0)
		
		denom = modulation_func(E_total / E_min)
		if denom == 0: denom = 1e-9

		for i in range(1, self.reso_sample_num - 1):
			self.reso_level_significance.append((E_total - E_min) * i / (self.reso_sample_num - 1) + E_min)
			self.reso_scales.append(scale_solver(scene_freq_image, self.reso_level_significance[-1]))
			self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
			self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / denom))
			
		self.reso_level_significance.append(E_total)
		self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
		self.reso_scales.append(1.)
		self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / denom))
		self.reso_level_begin.append(self.increase_reso_until)

		print(f"[ INFO ] Resolution scheduler initialized with {len(self.reso_scales)} levels")
		print(f"[ INFO ] Scale range: {self.max_reso_scale:.1f} → 1.0")