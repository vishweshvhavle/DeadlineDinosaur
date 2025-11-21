# Copyright (c) 2025 Harbin Institute of Technology, Huawei Noah's Ark Lab
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import math
import torch
from .. import arguments

class TrainingScheduler():
	"""
	DashGaussian training scheduler of resolution and primitive number.
	Implements momentum-based primitive budgeting.
	"""
	def __init__(self, opt: arguments.OptimizationParams, dens: arguments.DensifyParams, pipe: arguments.PipelineParams, init_n_gaussian: int, original_images: list) -> None:

		self.max_steps = opt.iterations
		self.init_n_gaussian = init_n_gaussian

		self.densify_mode = pipe.densify_mode
		self.densify_until_iter = dens.densify_until
		self.densification_interval = dens.densification_interval
		
		self.resolution_mode = pipe.resolution_mode

		self.start_significance_factor = 4
		self.max_reso_scale = 5
		self.reso_sample_num = 32
		self.max_densify_rate_per_step = 0.2
		self.reso_scales = None
		self.reso_level_significance = None
		self.reso_level_begin = None
		self.increase_reso_until = self.densify_until_iter
		self.next_i = 2

		# Momentum-based primitive budgeting (DashGaussian Eq. 5)
		# γ = 0.98, η = 1.0 (defaults from paper)
		self.integrate_factor = 0.98
		self.momentum_step_cap = 1000000
		
		if pipe.max_n_gaussian > 0:
			# Fixed mode: target is specified
			self.max_n_gaussian = pipe.max_n_gaussian
			self.momentum = -1  # Disable momentum
			print(f"[SCHEDULER] Fixed mode: target={self.max_n_gaussian} primitives")
		else:
			# Dynamic mode: organic growth targeting 1-2 million
			# Start with modest initial momentum for organic growth
			self.momentum = 2 * self.init_n_gaussian  # Reduced initial momentum
			self.max_n_gaussian = self.init_n_gaussian + self.momentum
			print(f"[SCHEDULER] Organic growth mode: P_fin_init={self.max_n_gaussian} (init={self.init_n_gaussian}, momentum={self.momentum}), γ={self.integrate_factor}")
		print(f"[SCHEDULER] Target range: 1-2 million primitives (organic)")
		# Generate schedulers
		self.init_reso_scheduler(original_images)
	
	def update_momentum(self, momentum_step, current_iteration=None, current_scale=None):
		"""
		Update momentum-based primitive budget to organically reach 1-2 million.
		"""
		if self.momentum == -1:
			# Fixed mode: no momentum update
			return
		
		# Apply cap to prevent explosive growth from single densification
		capped_step = min(self.momentum_step_cap, momentum_step)
		
		# Adaptive η that naturally slows down as we approach target range
		current_total = self.init_n_gaussian + self.momentum
		
		# Progress-based η reduction - becomes more conservative as we grow
		if current_total < 500000:
			# Early stage: moderate growth to reach 500K
			base_eta = 0.6
		elif current_total < 1000000:
			# Mid stage: slower growth to reach 1M
			base_eta = 0.4
		elif current_total < 1500000:
			# Late stage: very slow growth to approach 1.5M
			base_eta = 0.2
		else:
			# Final stage: minimal growth beyond 1.5M
			base_eta = 0.1
		
		# Scale-based adjustment
		if current_scale is not None and current_scale > 1:
			scale_factor = max(1.0, self.max_reso_scale / current_scale)
			eta = base_eta * (scale_factor ** 0.3)  # Very gentle scale dependence
			eta = min(eta, base_eta * 1.5)  # Limited boost from scale
		else:
			eta = base_eta
		
		# Time-based damping - become more conservative over iterations
		if current_iteration is not None:
			progress = min(current_iteration / self.densify_until_iter, 1.0)
			time_damping = 1.0 - (progress * 0.6)  # Reduce η by up to 60% over time
			eta *= time_damping
		
		# Size-based damping - natural slowdown as we grow larger
		size_damping = max(0.3, 1.0 - (current_total / 2000000))  # Dampens to 30% at 2M
		eta *= size_damping
		
		# DashGaussian Eq. 5 with multi-factor adaptive η
		old_momentum = self.momentum
		new_momentum = int(self.integrate_factor * self.momentum + eta * capped_step)
		self.momentum = max(self.momentum, new_momentum)
		
		# Update max_n_gaussian (P_fin)
		old_max = self.max_n_gaussian
		self.max_n_gaussian = self.init_n_gaussian + self.momentum
		
		if momentum_step > 0:
			eta_str = f", η={eta:.3f}" if current_scale is not None else ""
			scale_str = f", scale={current_scale:.0f}" if current_scale is not None else ""
			progress_str = f", progress={progress:.2f}" if current_iteration is not None else ""
			current_str = f", current={current_total}"
			print(f"[MOMENTUM] P_add={momentum_step}{eta_str}{scale_str}{progress_str}{current_str}, P_fin: {old_max} -> {self.max_n_gaussian} (+{self.max_n_gaussian - old_max})")
			
	def get_res_scale(self, iteration):
		"""Get current resolution scale based on iteration."""
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
		"""Calculate densification rate for organic 1-2M growth."""
		if self.densify_mode == "free":
			# Progressive reduction based on current size
			if cur_n_gaussian < 500000:
				return 0.4
			elif cur_n_gaussian < 1000000:
				return 0.2
			elif cur_n_gaussian < 1500000:
				return 0.1
			else:
				return 0.05
		elif self.densify_mode == "freq":
			assert cur_scale is not None, "cur_scale required for freq mode"
			
			progress = min(iteration / self.densify_until_iter, 1.0)
			
			# Adaptive power factor based on current size
			if cur_n_gaussian < 500000:
				power_factor = 2.0 - progress  # Standard growth
			elif cur_n_gaussian < 1000000:
				power_factor = 2.3 - progress  # Slower growth
			else:
				power_factor = 2.8 - progress  # Very slow growth
			
			# Target primitive count
			denominator = cur_scale ** power_factor
			target_n_gaussian = self.init_n_gaussian + (self.max_n_gaussian - self.init_n_gaussian) / denominator
			
			# Calculate remaining densification steps
			remaining_iters = self.densify_until_iter - iteration
			if remaining_iters <= 0:
				return 0.0
			
			remaining_steps = max(1, remaining_iters // self.densification_interval)
			
			# Conservative growth per step with size-based limiting
			total_to_add = max(0, target_n_gaussian - cur_n_gaussian)
			
			# Size-based step limiting
			max_step_size = min(50000, cur_n_gaussian * 0.1)  # Never add more than 10% of current size
			total_to_add = min(total_to_add, max_step_size)
			
			per_step_add = total_to_add / remaining_steps
			
			# Convert to rate
			densify_rate = per_step_add / self.init_n_gaussian
			
			# Adaptive clamp based on current size
			if cur_n_gaussian < 500000:
				max_rate = self.max_densify_rate_per_step * 0.5
			elif cur_n_gaussian < 1000000:
				max_rate = self.max_densify_rate_per_step * 0.3
			else:
				max_rate = self.max_densify_rate_per_step * 0.15
			
			densify_rate = max(0, min(densify_rate, max_rate))
			
			return densify_rate
	
	def lr_decay_from_iter(self):
		"""Determine when learning rate decay should start."""
		if self.resolution_mode == "const":
			return 1
		
		# Start decay when resolution scale drops below 2
		for i, s in zip(self.reso_level_begin, self.reso_scales):
			if s < 2:
				return i
		
		return self.increase_reso_until

	def init_reso_scheduler(self, original_images):
		"""Initialize frequency-based resolution scheduler using FFT analysis."""
		if self.resolution_mode != "freq":
			print(f"[INFO] Skipped resolution scheduler init, mode is {self.resolution_mode}")
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
		
		print("[INFO] Initializing resolution scheduler...")

		self.max_reso_scale = 8
		self.next_i = 2
		scene_freq_image = None
		
		for img_tensor in original_images:
			if img_tensor.dim() == 4:
				img_tensor = img_tensor.squeeze(0)
			if img_tensor.dim() != 3:
				raise ValueError(f"Image tensor has wrong dimensions: {img_tensor.shape}")
			
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
		if denom == 0:
			denom = 1e-9

		for i in range(1, self.reso_sample_num - 1):
			self.reso_level_significance.append((E_total - E_min) * (i - 0) / (self.reso_sample_num-1 - 0) + E_min)
			self.reso_scales.append(scale_solver(scene_freq_image, self.reso_level_significance[-1]))
			self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
			self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / denom))
			
		self.reso_level_significance.append(E_total)
		self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
		self.reso_scales.append(1.)
		self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / denom))
		self.reso_level_begin.append(self.increase_reso_until)

		print(f"[INFO] Resolution scheduler: {len(self.reso_scales)} levels, max_scale={self.max_reso_scale:.2f}")