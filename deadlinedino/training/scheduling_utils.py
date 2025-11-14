import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import time
import math
import sys
import os


class TrainingScheduler:
    """
    DashGaussian training scheduler with momentum-based primitive budgeting
    and frequency-aware resolution scheduling.

    This scheduler uses iteration-based progression (not time-based) and analyzes
    the frequency content of training images to determine optimal resolution schedules.
    """

    def __init__(self, opt, pipe, gaussians, original_images: list, densify_params=None) -> None:
        """
        Initialize the training scheduler.

        Args:
            opt: OptimizationParams containing training parameters
            pipe: PipelineParams containing pipeline configuration
            gaussians: GaussianModel instance
            original_images: List of training images for FFT analysis
            densify_params: Optional DensifyParams for densification interval (defaults to opt if not provided)
        """
        self.max_steps = opt.iterations
        self.init_n_gaussian = gaussians.get_xyz.shape[0]

        # Densification parameters
        self.densify_mode = getattr(pipe, 'densify_mode', 'freq')
        densify_until_iter = getattr(opt, 'densify_until_iter', -1)
        self.densify_until_iter = opt.iterations // 2 if densify_until_iter == -1 else densify_until_iter
        # Try densify_params first, fall back to opt, then to default
        if densify_params is not None:
            self.densification_interval = getattr(densify_params, 'densification_interval', 100)
        else:
            self.densification_interval = getattr(opt, 'densification_interval', 100)

        # Resolution parameters
        self.resolution_mode = getattr(pipe, 'resolution_mode', 'freq')

        # Resolution scheduler parameters
        self.start_significance_factor = 4
        self.max_reso_scale = 8
        self.reso_sample_num = 32  # Must be no less than 2
        self.max_densify_rate_per_step = 0.2
        self.reso_scales = None
        self.reso_level_significance = None
        self.reso_level_begin = None
        self.increase_reso_until = self.densify_until_iter
        self.next_i = 2

        # Momentum-based primitive budgeting
        if hasattr(pipe, 'max_n_gaussian') and pipe.max_n_gaussian > 0:
            self.max_n_gaussian = pipe.max_n_gaussian
            self.momentum = -1
        else:
            self.momentum = 5 * self.init_n_gaussian
            self.max_n_gaussian = self.init_n_gaussian + self.momentum
            self.integrate_factor = 0.98
            self.momentum_step_cap = 1000000

        # Initialize resolution scheduler
        self.init_reso_scheduler(original_images)

        # Store learning rate parameters for scheduler
        if self.resolution_mode == "freq":
            self.position_lr_init = opt.position_lr_init
            self.position_lr_final = opt.position_lr_final
            self.position_lr_delay_mult = getattr(opt, 'position_lr_delay_mult', 0.01)
            self.position_lr_max_steps = opt.position_lr_max_steps

    def update_momentum(self, momentum_step):
        """
        Update momentum for primitive budgeting.

        The momentum smoothly tracks the densification rate and helps prevent
        sudden spikes in primitive count.

        Args:
            momentum_step: Number of primitives added in the last densification step
        """
        if self.momentum == -1:
            return
        self.momentum = max(
            self.momentum,
            int(self.integrate_factor * self.momentum + min(self.momentum_step_cap, momentum_step))
        )
        self.max_n_gaussian = self.init_n_gaussian + self.momentum

    def get_res_scale(self, iteration):
        """
        Get the resolution scale for the current iteration.

        Args:
            iteration: Current training iteration

        Returns:
            Resolution scale factor (e.g., 8 for 1/8 resolution, 1 for full resolution)
        """
        if self.resolution_mode == "const":
            return 1
        elif self.resolution_mode == "freq":
            if iteration >= self.increase_reso_until:
                return 1
            if iteration < self.reso_level_begin[1]:
                return self.reso_scales[0]
            while iteration >= self.reso_level_begin[self.next_i]:
                # If the index is out of range, something is wrong with the scheduler
                self.next_i += 1
            i = self.next_i - 1
            i_now, i_nxt = self.reso_level_begin[i: i + 2]
            s_lst, s_now = self.reso_scales[i - 1: i + 1]
            # Smooth interpolation in frequency space
            scale = (1 / ((iteration - i_now) / (i_nxt - i_now) * (1/s_now**2 - 1/s_lst**2) + 1/s_lst**2))**0.5
            return int(scale)
        else:
            raise NotImplementedError("Resolution mode '{}' is not implemented.".format(self.resolution_mode))

    def get_densify_rate(self, iteration, cur_n_gaussian, cur_scale=None):
        """
        Get the maximum densification rate for the current iteration.

        This controls how many primitives can be added in a single densification step
        relative to the current primitive count.

        Args:
            iteration: Current training iteration
            cur_n_gaussian: Current number of Gaussian primitives
            cur_scale: Current resolution scale (required for 'freq' mode)

        Returns:
            Maximum densification rate (e.g., 0.2 = can add up to 20% more primitives)
        """
        if self.densify_mode == "free":
            return 1.0
        elif self.densify_mode == "freq":
            assert cur_scale is not None
            if self.densification_interval + iteration < self.increase_reso_until:
                next_n_gaussian = int(
                    (self.max_n_gaussian - self.init_n_gaussian) /
                    cur_scale**(2 - iteration / self.densify_until_iter)
                ) + self.init_n_gaussian
            else:
                next_n_gaussian = self.max_n_gaussian
            return min(
                max((next_n_gaussian - cur_n_gaussian) / cur_n_gaussian, 0.),
                self.max_densify_rate_per_step
            )
        else:
            raise NotImplementedError("Densify mode '{}' is not implemented.".format(self.densify_mode))

    def lr_decay_from_iter(self):
        """
        Determine from which iteration to start learning rate decay.

        Returns:
            Iteration number to start LR decay
        """
        if self.resolution_mode == "const":
            return 1
        for i, s in zip(self.reso_level_begin, self.reso_scales):
            if s < 2:
                return i
        raise Exception("Something is wrong with resolution scheduler.")

    def init_reso_scheduler(self, original_images):
        if self.resolution_mode != "freq":
            print("[ INFO ] Skipped resolution scheduler initialization, the resolution mode is {}".format(self.resolution_mode))
            return

        def compute_win_significance(significance_map: torch.Tensor, scale: float):
            h, w = significance_map.shape[-2:]
            c = ((h + 1) // 2, (w + 1) // 2)
            win_size = (int(h / scale), int(w / scale))
            win_significance = significance_map[..., c[0]-win_size[0]//2: c[0]+win_size[0]//2, c[1]-win_size[1]//2: c[1]+win_size[1]//2].sum().item()
            return win_significance
        
        def scale_solver(significance_map: torch.Tensor, target_significance: float):
            L, R, T = 0., 1., 64
            for _ in range(T):
                mid = (L + R) / 2
                win_significance = compute_win_significance(significance_map, 1 / mid)
                if win_significance < target_significance:
                    L = mid
                else:
                    R = mid
            return 1 / mid
        
        print("[ INFO ] Initializing resolution scheduler...")

        self.max_reso_scale = 8
        self.next_i = 2
        scene_freq_image = None
        
        # Calculate high-frequency ratio for threshold decision
        total_high_freq_energy = 0.0
        total_energy = 0.0
        
        for img in original_images:
            img_fft_centered = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
            img_fft_centered_mod = (img_fft_centered.real.square() + img_fft_centered.imag.square()).sqrt()
            scene_freq_image = img_fft_centered_mod if scene_freq_image is None else scene_freq_image + img_fft_centered_mod

            # Calculate high-frequency ratio for this image
            h, w = img_fft_centered_mod.shape[-2:]
            center_h, center_w = h // 2, w // 2
            
            # High frequencies: outer 50%
            high_freq_radius_h = int(h * 0.25)
            high_freq_radius_w = int(w * 0.25)
            
            high_freq_mask = torch.ones_like(img_fft_centered_mod)
            high_freq_mask[center_h - high_freq_radius_h:center_h + high_freq_radius_h,
                        center_w - high_freq_radius_w:center_w + high_freq_radius_w] = 0
            
            high_freq_energy = (img_fft_centered_mod * high_freq_mask).sum().item()
            img_total_energy = img_fft_centered_mod.sum().item()
            
            total_high_freq_energy += high_freq_energy
            total_energy += img_total_energy

            e_total = img_fft_centered_mod.sum().item()
            e_min = e_total / self.start_significance_factor
            self.max_reso_scale = min(self.max_reso_scale, scale_solver(img_fft_centered_mod, e_min))

        # Calculate overall high-frequency ratio
        high_freq_ratio = total_high_freq_energy / total_energy if total_energy > 0 else 0
        print(f"[ INFO ] Scene high-frequency ratio: {high_freq_ratio:.4f}")

        # DYNAMIC SCALE SELECTION BASED ON 0.12 THRESHOLD
        if high_freq_ratio > 0.12:
            # High-frequency scene: start at 1/5 scale for more aggressive compression
            starting_scale = 5.0
            print(f"[ INFO ] High-frequency scene detected (>0.12), starting at 1/5 scale")
        else:
            # Low-frequency scene: start at 1/4 scale for faster training
            starting_scale = 4.0
            print(f"[ INFO ] Low-frequency scene detected (≤0.12), starting at 1/4 scale")

        modulation_func = math.log

        self.reso_scales = []
        self.reso_level_significance = []
        self.reso_level_begin = []
        scene_freq_image /= len(original_images)
        E_total = scene_freq_image.sum().item()
        E_min = compute_win_significance(scene_freq_image, self.max_reso_scale)
        
        # Override the first scale with our dynamic choice
        self.reso_level_significance.append(E_min)
        self.reso_scales.append(starting_scale)  # Use dynamic starting scale
        self.reso_level_begin.append(0)
        
        # Generate intermediate scales
        for i in range(1, self.reso_sample_num - 1):
            self.reso_level_significance.append((E_total - E_min) * (i - 0) / (self.reso_sample_num-1 - 0) + E_min)
            self.reso_scales.append(scale_solver(scene_freq_image, self.reso_level_significance[-1]))
            self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
            self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / modulation_func(E_total / E_min)))
        
        # Final full resolution
        self.reso_level_significance.append(modulation_func(E_total / E_min))
        self.reso_scales.append(1.)
        self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
        self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / modulation_func(E_total / E_min)))
        self.reso_level_begin.append(self.increase_reso_until)

        # ADAPTIVE PROGRESSION BASED ON FREQUENCY CONTENT
        # High-frequency scenes progress slower through scales
        if high_freq_ratio > 0.12:
            # For high-frequency scenes, spend more time at each scale
            # Adjust the progression to be more gradual
            scale_adjustment_factor = 0.7  # Slower progression
        else:
            # For low-frequency scenes, progress faster
            scale_adjustment_factor = 1.3  # Faster progression
        
        # Apply non-linear adjustment to the progression
        adjusted_begin = []
        for i, begin_iter in enumerate(self.reso_level_begin):
            if i == 0:
                adjusted_begin.append(0)
            elif i == len(self.reso_level_begin) - 1:
                adjusted_begin.append(self.increase_reso_until)
            else:
                # Apply power law adjustment
                progress = begin_iter / self.increase_reso_until
                adjusted_progress = progress ** scale_adjustment_factor
                adjusted_begin.append(int(adjusted_progress * self.increase_reso_until))
        
        self.reso_level_begin = adjusted_begin

        print(f"[ INFO ] Resolution scheduler initialized with {len(self.reso_scales)} levels")
        print(f"[ INFO ] Starting scale: 1/{starting_scale:.1f}")
        print(f"[ INFO ] Max resolution scale: {self.max_reso_scale:.2f}")
        print(f"[ INFO ] Resolution will increase until iteration: {self.increase_reso_until}")

    def get_resolution_scale_inverse(self, iteration):
        """
        Get the inverse resolution scale (for compatibility with existing code).

        Args:
            iteration: Current training iteration

        Returns:
            1 / scale (e.g., 1/8 for scale=8, 1 for scale=1)
        """
        scale = self.get_res_scale(iteration)
        return 1.0 / scale
    
    def adapt_resolution_progression(self, current_iteration: int, training_metrics: dict = None):
        """
        Dynamically adjust resolution progression based on training progress.
        
        This method can be called during training to adapt the resolution schedule
        based on how well the training is progressing.
        
        Args:
            current_iteration: Current training iteration
            training_metrics: Dictionary containing training metrics like:
                - 'psnr': Current PSNR value
                - 'loss': Current loss value
                - 'convergence_rate': Rate of improvement
        """
        if self.resolution_mode != "freq" or not self.reso_level_begin:
            return
        
        # Only adapt after we've progressed through at least 2 scales
        if current_iteration < self.reso_level_begin[2]:
            return
        
        # Calculate current progress
        progress_ratio = current_iteration / self.increase_reso_until
        
        # If we have training metrics, use them to guide adaptation
        if training_metrics:
            psnr = training_metrics.get('psnr', 0)
            loss = training_metrics.get('loss', float('inf'))
            
            # If training is going well (high PSNR or low loss), consider accelerating
            if psnr > 25 or loss < 0.1:
                # Training is going well - consider faster progression
                acceleration_factor = 1.1
                self._accelerate_progression(acceleration_factor, current_iteration)
            elif psnr < 20 and loss > 0.5:
                # Training is struggling - consider slowing down
                deceleration_factor = 0.9
                self._decelerate_progression(deceleration_factor, current_iteration)

    def _accelerate_progression(self, factor: float, current_iteration: int):
        """Accelerate resolution progression by the given factor."""
        # Only adjust future scale transitions
        for i in range(len(self.reso_level_begin)):
            if self.reso_level_begin[i] > current_iteration:
                remaining_iterations = self.reso_level_begin[i] - current_iteration
                accelerated_remaining = int(remaining_iterations * factor)
                self.reso_level_begin[i] = current_iteration + accelerated_remaining

    def _decelerate_progression(self, factor: float, current_iteration: int):
        """Decelerate resolution progression by the given factor."""
        # Only adjust future scale transitions
        for i in range(len(self.reso_level_begin)):
            if self.reso_level_begin[i] > current_iteration:
                remaining_iterations = self.reso_level_begin[i] - current_iteration
                decelerated_remaining = int(remaining_iterations * factor)
                self.reso_level_begin[i] = current_iteration + decelerated_remaining

class Scheduler(_LRScheduler):
    """
    Exponential learning rate scheduler with log-linear interpolation.
    Only applies to the 'xyz' (position) parameter group.
    """
    def __init__(self, optimizer: torch.optim.Adam, lr_init, lr_final, max_epochs=10000, last_epoch=-1):
        self.max_epochs = max_epochs
        self.lr_init = lr_init
        self.lr_final = lr_final
        super(Scheduler, self).__init__(optimizer, last_epoch)
        return

    def __helper(self):
        if self.last_epoch < 0 or (self.lr_init == 0.0 and self.lr_final == 0.0):
            # Disable this parameter
            return 0.0
        delay_rate = 1.0
        t = np.clip(self.last_epoch / self.max_epochs, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return delay_rate * log_lerp

    def get_lr(self):
        lr_list = []
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                lr_list.append(self.__helper())
            else:
                lr_list.append(group['initial_lr'])

        return lr_list

class ResolutionScheduler:
    """
    Frequency-aware resolution scheduler that progressively increases resolution during training.

    Based on DashGaussian implementation with FFT-based frequency analysis.
    Uses iteration-based progression instead of time-based.
    """

    def __init__(self, opt, pipe, original_images: list = None, 
                tile_height: int = 8, tile_width: int = 16):
        """
        Initialize the resolution scheduler.

        Args:
            opt: OptimizationParams with training parameters
            pipe: PipelineParams with pipeline configuration  
            original_images: List of training images for FFT analysis
            tile_height: Tile height for CUDA kernels
            tile_width: Tile width for CUDA kernels
        """
        self.opt = opt
        self.pipe = pipe
        self.tile_height = tile_height
        self.tile_width = tile_width
        
        # Resolution parameters from DashGaussian
        self.resolution_mode = getattr(pipe, 'resolution_mode', 'freq')
        self.start_significance_factor = 4
        self.max_reso_scale = 8
        self.reso_sample_num = 32
        self.reso_scales = None
        self.reso_level_significance = None
        self.reso_level_begin = None
        
        # FIX: Ensure increase_reso_until is valid BEFORE initializing scheduler
        densify_until = getattr(opt, 'densify_until_iter', -1)
        if densify_until <= 0:
            self.increase_reso_until = opt.iterations // 2
        else:
            self.increase_reso_until = densify_until
        
        # Ensure we have a valid value
        if self.increase_reso_until <= 0:
            self.increase_reso_until = max(1, opt.iterations // 2)
            
        print(f"[ DEBUG ] opt.iterations={opt.iterations}, increase_reso_until={self.increase_reso_until}")
        
        self.next_i = 2
        self.current_iteration = 0

        # Initialize based on mode
        if self.resolution_mode == "freq" and original_images is not None and len(original_images) > 0:
            try:
                self._init_fft_scheduler(original_images)
                print("[ INFO ] Successfully initialized FFT-based resolution scheduler")
            except Exception as e:
                print(f"[ WARNING ] FFT scheduler initialization failed: {e}")
                print("[ INFO ] Falling back to paper-style scheduler")
                self._init_simple_scheduler()
        elif self.resolution_mode == "const":
            self.reso_scales = [1]
            self.reso_level_begin = [0]
            print("[ INFO ] Using constant resolution (full res)")
        else:
            # Fallback to simple progression
            self._init_simple_scheduler()
            print("[ INFO ] Using paper-style resolution scheduler")

    def _init_fft_scheduler(self, images: list):
        """Initialize resolution scales based on FFT analysis of training images.
        
        Uses FFT to determine how quickly to ramp up from 1/5 resolution to full resolution
        based on high-frequency content in the scene.
        """
        print("[ INFO ] Initializing FFT-based resolution scheduler...")

        scene_freq_image = None
        total_high_freq_energy = 0.0
        total_energy = 0.0

        # Analyze frequency content of all images
        for img_idx, img in enumerate(images):
            # Convert to torch tensor if needed
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)

            # Ensure image is on GPU for faster FFT
            if torch.cuda.is_available() and not img.is_cuda:
                img = img.cuda()

            # Ensure image is float and in correct format
            if img.dtype == torch.uint8:
                img = img.float() / 255.0

            # Handle different tensor formats
            if len(img.shape) == 3:  # (H, W, C) or (C, H, W)
                if img.shape[0] in [1, 3]:  # (C, H, W)
                    img = img.permute(1, 2, 0)
                # Convert to grayscale for frequency analysis
                if img.shape[2] == 3:
                    img_gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
                else:
                    img_gray = img[..., 0]
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")

            # Compute FFT
            try:
                img_fft = torch.fft.fft2(img_gray)
                img_fft_centered = torch.fft.fftshift(img_fft, dim=(-2, -1))
                img_fft_magnitude = torch.sqrt(img_fft_centered.real.pow(2) + img_fft_centered.imag.pow(2))
                
                # Accumulate frequency maps
                if scene_freq_image is None:
                    scene_freq_image = img_fft_magnitude
                else:
                    if scene_freq_image.shape == img_fft_magnitude.shape:
                        scene_freq_image = scene_freq_image + img_fft_magnitude
                    else:
                        print(f"[ WARNING ] Skipping image {img_idx} due to dimension mismatch")
                        continue

                # Compute high-frequency energy ratio
                h, w = img_fft_magnitude.shape
                center_h, center_w = h // 2, w // 2
                
                # Define high-frequency region: outer 40% of frequency space
                high_freq_radius_h = int(h * 0.3)
                high_freq_radius_w = int(w * 0.3)
                
                # Create mask for high frequencies (everything outside center region)
                mask = torch.ones_like(img_fft_magnitude)
                mask[center_h - high_freq_radius_h:center_h + high_freq_radius_h,
                     center_w - high_freq_radius_w:center_w + high_freq_radius_w] = 0
                
                high_freq_energy = (img_fft_magnitude * mask).sum().item()
                img_total_energy = img_fft_magnitude.sum().item()
                
                total_high_freq_energy += high_freq_energy
                total_energy += img_total_energy
                
            except Exception as e:
                print(f"[ WARNING ] FFT failed for image {img_idx}: {e}")
                continue

        if scene_freq_image is None or total_energy == 0:
            print("[ WARNING ] FFT analysis failed, falling back to simple scheduler")
            self._init_simple_scheduler()
            return

        # Compute high-frequency ratio for the scene
        high_freq_ratio = total_high_freq_energy / total_energy
        print(f"[ INFO ] Scene high-frequency ratio: {high_freq_ratio:.4f}")

        # Determine number of resolution stages based on high-frequency content
        # More high-freq content → more stages → slower ramp up
        # Less high-freq content → fewer stages → faster ramp up
        if high_freq_ratio > 0.3:
            # High detail scene: use 6 stages, ramp up slowly
            num_stages = 6
            print(f"[ INFO ] High-detail scene detected, using {num_stages} resolution stages")
        elif high_freq_ratio > 0.15:
            # Medium detail scene: use 5 stages
            num_stages = 5
            print(f"[ INFO ] Medium-detail scene detected, using {num_stages} resolution stages")
        else:
            # Low detail scene: use 4 stages, ramp up quickly
            num_stages = 4
            print(f"[ INFO ] Low-detail scene detected, using {num_stages} resolution stages")

        # Always start at 1/5 resolution (minimum supported)
        min_scale = 5.0
        
        # Create resolution progression: 1/5 → 1/4 → 1/3 → 1/2 → 1/1
        # For 6 stages: 1/5 → 1/4 → 1/3 → 1/2.5 → 1/2 → 1/1
        # For 5 stages: 1/5 → 1/4 → 1/3 → 1/2 → 1/1
        # For 4 stages: 1/5 → 1/3 → 1/2 → 1/1
        
        if num_stages == 6:
            self.reso_scales = [5.0, 4.0, 3.0, 2.5, 2.0, 1.0]
        elif num_stages == 5:
            self.reso_scales = [5.0, 4.0, 3.0, 2.0, 1.0]
        else:  # 4 stages
            self.reso_scales = [5.0, 3.0, 2.0, 1.0]

        # Distribute stages across training iterations
        # Use non-linear distribution: spend more time at lower resolutions for high-detail scenes
        self.reso_level_begin = []
        
        if high_freq_ratio > 0.3:
            # High detail: slower progression (more iterations at low res)
            # Use exponential spacing
            for i in range(num_stages):
                # Exponential spacing: spend more time at beginning
                progress = (i / (num_stages - 1)) ** 1.5  # Exponent > 1 → slower start
                begin_iter = int(self.increase_reso_until * progress)
                self.reso_level_begin.append(begin_iter)
        elif high_freq_ratio > 0.15:
            # Medium detail: linear progression
            for i in range(num_stages):
                begin_iter = int(self.increase_reso_until * i / (num_stages - 1))
                self.reso_level_begin.append(begin_iter)
        else:
            # Low detail: faster progression (less time at low res)
            # Use square root spacing
            for i in range(num_stages):
                progress = (i / (num_stages - 1)) ** 0.7  # Exponent < 1 → faster start
                begin_iter = int(self.increase_reso_until * progress)
                self.reso_level_begin.append(begin_iter)

        # Ensure first stage starts at 0
        self.reso_level_begin[0] = 0
        
        # Ensure scales are in descending order and begin times are increasing
        self.reso_scales = sorted(self.reso_scales, reverse=True)
        self.reso_level_begin = sorted(self.reso_level_begin)

        print(f"[ INFO ] FFT scheduler initialized with {len(self.reso_scales)} resolution stages")
        print(f"[ INFO ] Resolution scales: {[f'1/{s:.1f}' for s in self.reso_scales]}")
        print(f"[ INFO ] Stage begin iterations: {self.reso_level_begin}")
        print(f"[ INFO ] Resolution will increase until iteration: {self.increase_reso_until}")

    def _init_simple_scheduler(self):
        """Use paper's resolution progression: 1/5, 1/4, 1/3, 1/2, 1/1"""
        print("[ INFO ] Initializing paper-style resolution scheduler...")
        
        # Fix the increase_reso_until value if needed
        if self.increase_reso_until <= 0:
            self.increase_reso_until = max(1, self.opt.iterations // 2)
            print(f"[ INFO ] Fixed increase_reso_until to: {self.increase_reso_until}")
        
        # Paper's progression: 1/5, 1/4, 1/3, 1/2, 1/1 (larger minimum resolution)
        self.reso_scales = [5, 4, 3, 2, 1]  # Inverse of the scale factors
        num_stages = len(self.reso_scales)
        
        # Distribute stages evenly across training
        self.reso_level_begin = []
        for i in range(num_stages):
            begin_iter = int(self.increase_reso_until * i / (num_stages - 1))
            self.reso_level_begin.append(begin_iter)
        
        print(f"[ INFO ] Paper scheduler: scales {self.reso_scales}, begins {self.reso_level_begin}")

    def step(self, iteration: int):
        """Update the current iteration."""
        self.current_iteration = iteration

    def get_resolution_scale(self) -> float:
        """Get the current resolution scale factor."""
        if self.resolution_mode == "const":
            return 1.0
        
        iteration = self.current_iteration
        
        if iteration >= self.increase_reso_until:
            return 1.0
        
        if len(self.reso_level_begin) < 2 or iteration < self.reso_level_begin[1]:
            return 1.0 / self.reso_scales[0]
        
        # Update next_i to current position
        while (self.next_i < len(self.reso_level_begin) and 
               iteration >= self.reso_level_begin[self.next_i]):
            self.next_i += 1

        if self.next_i >= len(self.reso_level_begin):
            return 1.0  # Full resolution

        # Interpolate between resolution levels (DashGaussian style)
        i = self.next_i - 1
        if i + 1 >= len(self.reso_level_begin):
            return 1.0
            
        i_now, i_nxt = self.reso_level_begin[i: i + 2]
        if i < 1 or i >= len(self.reso_scales):
            return 1.0
            
        s_lst, s_now = self.reso_scales[i - 1: i + 1]

        # Smooth interpolation in frequency space
        if i_nxt == i_now:  # Avoid division by zero
            return 1.0 / s_now
            
        scale = (1 / ((iteration - i_now) / (i_nxt - i_now) * (1/s_now**2 - 1/s_lst**2) + 1/s_lst**2))**0.5
        return 1.0 / scale

    def get_downsampled_dimensions(self, full_height: int, full_width: int) -> tuple[int, int]:
        """
        Get the downsampled image shape based on current resolution scale.
        Ensures dimensions are compatible with CUDA tile-based rendering.
        """
        scale = self.get_resolution_scale()
        downsampled_height = int(full_height * scale)
        downsampled_width = int(full_width * scale)

        # Ensure minimum dimensions: at least 4x the tile size for safety
        min_height = self.tile_height * 4  # Increased from 2x to 4x
        min_width = self.tile_width * 4    # Increased from 2x to 4x
        downsampled_height = max(min_height, downsampled_height)
        downsampled_width = max(min_width, downsampled_width)

        # Round to nearest multiple of tile size for better alignment
        downsampled_height = ((downsampled_height + self.tile_height - 1) // self.tile_height) * self.tile_height
        downsampled_width = ((downsampled_width + self.tile_width - 1) // self.tile_width) * self.tile_width

        # Additional safety: ensure we have enough tiles in both dimensions
        # At least 16 tiles in width and 12 tiles in height
        min_tiles_h = 12
        min_tiles_w = 16
        downsampled_height = max(downsampled_height, self.tile_height * min_tiles_h)
        downsampled_width = max(downsampled_width, self.tile_width * min_tiles_w)

        # Ensure we don't exceed original dimensions
        downsampled_height = min(downsampled_height, full_height)
        downsampled_width = min(downsampled_width, full_width)

        return downsampled_height, downsampled_width

    def get_downsampled_proj_matrix(self, proj_matrix: np.ndarray,
                                     full_height: int, full_width: int) -> np.ndarray:
        """
        Get a projection matrix adjusted for rendering at reduced resolution.

        Args:
            proj_matrix: Original projection matrix (4x4)
            full_height: Full resolution height
            full_width: Full resolution width

        Returns:
            Adjusted projection matrix for rendering at downsampled resolution
        """
        scale = self.get_resolution_scale()
        downsampled_proj = proj_matrix.copy()
        
        # Scale principal points
        if abs(proj_matrix[0, 2]) > 1e-6:
            downsampled_proj[0, 2] = proj_matrix[0, 2] * scale
        if abs(proj_matrix[1, 2]) > 1e-6:
            downsampled_proj[1, 2] = proj_matrix[1, 2] * scale
        
        return downsampled_proj

    @staticmethod
    def downsample_image_hq(image: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """
        High-quality image downsampling using torch interpolation.

        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            target_height: Target height
            target_width: Target width

        Returns:
            Downsampled image tensor
        """
        if image.shape[-2] == target_height and image.shape[-1] == target_width:
            return image

        original_shape = image.shape
        batch_mode = len(original_shape) == 4

        if not batch_mode:
            image = image.unsqueeze(0)

        # Use area interpolation for downsampling
        downsampled = torch.nn.functional.interpolate(
            image,
            size=(target_height, target_width),
            mode='area'
        )

        if not batch_mode:
            downsampled = downsampled.squeeze(0)

        return downsampled

    def get_info_dict(self) -> dict:
        """
        Get information about the current scheduler state.

        Returns:
            Dictionary with iteration, scale, and resolution info
        """
        scale = self.get_resolution_scale()
        current_scale = 1.0 / scale if scale > 0 else 1.0

        return {
            'iteration': self.current_iteration,
            'scale': scale,
            'current_scale': current_scale,
            'max_iterations': self.opt.iterations,
            'increase_until': self.increase_reso_until
        }