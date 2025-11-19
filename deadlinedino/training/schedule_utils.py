# Copyright (c) 2025 Harbin Institute of Technology, Huawei Noah's Ark Lab
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import math
import torch
from .. import arguments

class TrainingScheduler():
	"""
	DashGaussian training scheduler of resolution and primitive number.
	"""
 	# --- MODIFIED: Added 'dens: arguments.DensifyParams' ---
	def __init__(self, opt: arguments.OptimizationParams, dens: arguments.DensifyParams, pipe: arguments.PipelineParams, init_n_gaussian: int, original_images: list) -> None:

		self.max_steps = opt.iterations
		self.init_n_gaussian = init_n_gaussian

		self.densify_mode = pipe.densify_mode
  
		# --- FIXED: Corrected attribute name ---
		self.densify_until_iter = dens.densify_until # Changed from densify_until_iter
		# --- END FIX ---
		self.densification_interval = dens.densification_interval
		

		self.resolution_mode = pipe.resolution_mode

		self.start_significance_factor = 4
		self.max_reso_scale = 8
		self.reso_sample_num = 32 # Must be no less than 2
		self.max_densify_rate_per_step = 0.2
		self.reso_scales = None
		self.reso_level_significance = None
		self.reso_level_begin = None
		self.increase_reso_until = self.densify_until_iter
		self.next_i = 2

		if pipe.max_n_gaussian > 0:
			self.max_n_gaussian = pipe.max_n_gaussian
			self.momentum = -1
		else:
			self.momentum = 5 * self.init_n_gaussian
			self.max_n_gaussian = self.init_n_gaussian + self.momentum
			self.integrate_factor = 0.98
			self.momentum_step_cap = 1000000
		
		# Generate schedulers.
		self.init_reso_scheduler(original_images)

		# NOTE: We DO NOT modify the optimizer here.
		# The trainer will call lr_decay_from_iter() and pass it to the optimizer.
	
	def update_momentum(self, momentum_step):
		if self.momentum == -1:
			return
		self.momentum = max(self.momentum, 
					        int(self.integrate_factor * self.momentum + min(self.momentum_step_cap, momentum_step)))
		self.max_n_gaussian = self.init_n_gaussian + self.momentum

	def get_res_scale(self, iteration):

		#DEBUG
		# print(f"resolution_mode: {self.resolution_mode}, iteration: {iteration}, increase_reso_until: {self.increase_reso_until}")

		if self.resolution_mode == "const":
			return 1
		elif self.resolution_mode == "freq":
			if iteration >= self.increase_reso_until:
				return 1
			if iteration < self.reso_level_begin[1]:
				return self.reso_scales[0]
			while iteration >= self.reso_level_begin[self.next_i]:
				# If the index is out of the range of 'reso_level_begin', there must be something wrong with the scheduler.
				self.next_i += 1
			i = self.next_i - 1
			i_now, i_nxt = self.reso_level_begin[i: i + 2]
			s_lst, s_now = self.reso_scales[i - 1: i + 1]
			scale = (1 / ((iteration - i_now) / (i_nxt - i_now) * (1/s_now**2 - 1/s_lst**2) + 1/s_lst**2))**0.5
			
			# Return the integer part of the scale, but ensure it's at least 1
			return max(1, int(scale))
		else:
			raise NotImplementedError("Resolution mode '{}' is not implemented.".format(self.resolution_mode))
	
	def get_densify_rate(self, iteration, cur_n_gaussian, cur_scale=None):
		if self.densify_mode == "free":
			return 1.0
		elif self.densify_mode == "freq":
			assert cur_scale is not None
			if self.densification_interval + iteration < self.increase_reso_until:
				next_n_gaussian = int((self.max_n_gaussian - self.init_n_gaussian) / cur_scale**(2 - iteration / self.densify_until_iter)) + self.init_n_gaussian
			else:
				next_n_gaussian = self.max_n_gaussian
			return min(max((next_n_gaussian - cur_n_gaussian) / cur_n_gaussian, 0.), self.max_densify_rate_per_step)
		else:
			raise NotImplementedError("Densify mode '{}' is not implemented.".format(self.densify_mode))
	
	def lr_decay_from_iter(self):
		if self.resolution_mode == "const":
			return 1 # Start decay right away
		
		# In freq mode, find the iteration where scale drops below 2
		for i, s in zip(self.reso_level_begin, self.reso_scales):
			if s < 2:
				return i # Start decay at this iteration
		
		# If it never drops below 2 (e.g., short training), start decay at the end
		return self.increase_reso_until

	def init_reso_scheduler(self, original_images):
		if self.resolution_mode != "freq":
			print("[ INFO ] Skipped resolution scheduler initialization, the resolution mode is {}".format(self.resolution_mode))
			return

		def compute_win_significance(significance_map: torch.Tensor, scale: float):
			# Ensure we are working with at least 3 dims (C, H, W)
			if significance_map.dim() < 3:
				significance_map = significance_map.unsqueeze(0)
			
			h, w = significance_map.shape[-2:]
			c = ((h + 1) // 2, (w + 1) // 2)
			win_size = (max(1, int(h / scale)), max(1, int(w / scale))) # Ensure window size is at least 1
			
			h_start = max(0, c[0] - win_size[0] // 2)
			h_end = min(h, c[0] + (win_size[0] + 1) // 2) # Adjust for odd sizes
			w_start = max(0, c[1] - win_size[1] // 2)
			w_end = min(w, c[1] + (win_size[1] + 1) // 2)

			win_significance = significance_map[..., h_start:h_end, w_start:w_end].sum().item()
			return win_significance
		
		def scale_solver(significance_map: torch.Tensor, target_significance: float):
			L, R, T = 0., 1., 64
			for _ in range(T):
				mid = (L + R) / 2
				if mid == 0: # Avoid division by zero if L/R are at 0
					break
				win_significance = compute_win_significance(significance_map, 1 / mid)
				if win_significance < target_significance:
					L = mid
				else:
					R = mid
			return 1 / max(L, 1e-9) # Return 1/L, ensuring L is not zero
		
		print("[ INFO ] Initializing resolution scheduler...")

		self.max_reso_scale = 8
		self.next_i = 2
		scene_freq_image = None
		
		for img_tensor in original_images:
			# Ensure image is (C, H, W) and float
			if img_tensor.dim() == 4: # [1, C, H, W]
				img_tensor = img_tensor.squeeze(0)
			if img_tensor.dim() != 3:
				raise ValueError(f"Image tensor has wrong dimensions: {img_tensor.shape}")
			
			img_tensor = img_tensor.float()
			
			# Use avg of color channels, or first channel if grayscale
			if img_tensor.shape[0] > 1:
				img = img_tensor.mean(dim=0) # [H, W]
			else:
				img = img_tensor.squeeze(0) # [H, W]

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
		
		# Ensure E_min is not zero to avoid log(0)
		E_min = max(E_min, 1e-9)
		E_total = max(E_total, E_min + 1e-9) # Ensure E_total > E_min
		
		self.reso_level_significance.append(E_min)
		self.reso_scales.append(self.max_reso_scale)
		self.reso_level_begin.append(0)
		
		denom = modulation_func(E_total / E_min)
		if denom == 0: denom = 1e-9 # Avoid division by zero

		for i in range(1, self.reso_sample_num - 1):
			self.reso_level_significance.append((E_total - E_min) * (i - 0) / (self.reso_sample_num-1 - 0) + E_min)
			self.reso_scales.append(scale_solver(scene_freq_image, self.reso_level_significance[-1]))
			self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
			self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / denom))
			
		self.reso_level_significance.append(E_total) # Add E_total
		self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
		self.reso_scales.append(1.)
		self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / denom))
		self.reso_level_begin.append(self.increase_reso_until)

		# print("================== Resolution Scheduler ==================")
		# for idx, (e, s, i) in enumerate(zip(self.reso_level_significance, self.reso_scales, self.reso_level_begin)):
		# 	print(" - idx: {:02d}; scale: {:.2f}; significance: {:.2f}; begin: {}".format(idx, s, e, i))
		# print("==========================================================")