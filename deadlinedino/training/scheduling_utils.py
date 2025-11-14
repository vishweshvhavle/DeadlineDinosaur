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
        """
        Initialize resolution scheduler based on FFT analysis of training images.

        This method analyzes the frequency content of images to determine optimal
        resolution scales. Images with more high-frequency content will maintain
        higher resolutions for longer during training.

        Args:
            original_images: List of training images (torch.Tensor or numpy arrays)
        """
        if self.resolution_mode != "freq":
            print("[ INFO ] Skipped resolution scheduler initialization, the resolution mode is {}".format(
                self.resolution_mode))
            return

        def compute_win_significance(significance_map: torch.Tensor, scale: float):
            """Compute frequency significance within a centered window."""
            h, w = significance_map.shape[-2:]
            c = ((h + 1) // 2, (w + 1) // 2)
            win_size = (int(h / scale), int(w / scale))
            win_significance = significance_map[
                ...,
                c[0] - win_size[0] // 2: c[0] + win_size[0] // 2,
                c[1] - win_size[1] // 2: c[1] + win_size[1] // 2
            ].sum().item()
            return win_significance

        def scale_solver(significance_map: torch.Tensor, target_significance: float):
            """Binary search to find scale that captures target frequency significance."""
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

        for img in original_images:
            # Convert to torch tensor if needed
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)

            # Ensure image is on GPU for faster FFT
            if not img.is_cuda:
                img = img.cuda()

            # Ensure image is float
            if img.dtype == torch.uint8:
                img = img.float() / 255.0

            # Compute FFT
            img_fft_centered = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
            img_fft_centered_mod = (img_fft_centered.real.square() + img_fft_centered.imag.square()).sqrt()
            scene_freq_image = img_fft_centered_mod if scene_freq_image is None else scene_freq_image + img_fft_centered_mod

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

        # First level (lowest resolution)
        self.reso_level_significance.append(E_min)
        self.reso_scales.append(self.max_reso_scale)
        self.reso_level_begin.append(0)

        # Intermediate levels
        for i in range(1, self.reso_sample_num - 1):
            self.reso_level_significance.append(
                (E_total - E_min) * (i - 0) / (self.reso_sample_num - 1 - 0) + E_min
            )
            self.reso_scales.append(scale_solver(scene_freq_image, self.reso_level_significance[-1]))
            self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
            self.reso_level_begin.append(
                int(self.increase_reso_until * self.reso_level_significance[-2] / modulation_func(E_total / E_min))
            )

        # Final level (full resolution)
        self.reso_level_significance.append(modulation_func(E_total / E_min))
        self.reso_scales.append(1.)
        self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
        self.reso_level_begin.append(
            int(self.increase_reso_until * self.reso_level_significance[-2] / modulation_func(E_total / E_min))
        )
        self.reso_level_begin.append(self.increase_reso_until)

        print(f"[ INFO ] Resolution scheduler initialized with {len(self.reso_scales)} levels")
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
        self.increase_reso_until = getattr(opt, 'densify_until_iter', opt.iterations // 2)
        self.next_i = 2
        self.current_iteration = 0

        # Initialize based on mode
        if self.resolution_mode == "freq" and original_images is not None:
            self._init_fft_scheduler(original_images)
        elif self.resolution_mode == "const":
            self.reso_scales = [1]
            self.reso_level_begin = [0]
        else:
            # Fallback to simple progression
            self._init_simple_scheduler()

    def _compute_win_significance(self, significance_map: torch.Tensor, scale: float) -> float:
        """Compute frequency significance within a centered window."""
        h, w = significance_map.shape[-2:]
        c = ((h + 1) // 2, (w + 1) // 2)
        win_size = (int(h / scale), int(w / scale))
        win_significance = significance_map[
            ...,
            c[0] - win_size[0] // 2: c[0] + win_size[0] // 2,
            c[1] - win_size[1] // 2: c[1] + win_size[1] // 2
        ].sum().item()
        return win_significance

    def _scale_solver(self, significance_map: torch.Tensor, target_significance: float) -> float:
        """Binary search to find scale that captures target frequency significance."""
        L, R, T = 0., 1., 64
        for _ in range(T):
            mid = (L + R) / 2
            win_significance = self._compute_win_significance(significance_map, 1 / mid)
            if win_significance < target_significance:
                L = mid
            else:
                R = mid
        return 1 / mid

    def _init_fft_scheduler(self, images: list):
        """Initialize resolution scales based on FFT analysis of training images."""
        print("[ INFO ] Initializing FFT-based resolution scheduler...")

        scene_freq_image = None

        # Analyze frequency content of all images
        for img in images:
            # Convert to torch tensor if needed
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)

            # Ensure image is on GPU for faster FFT
            if not img.is_cuda:
                img = img.cuda()

            # Ensure image is float
            if img.dtype == torch.uint8:
                img = img.float() / 255.0

            # Compute FFT
            img_fft_centered = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
            img_fft_centered_mod = (img_fft_centered.real.square() + img_fft_centered.imag.square()).sqrt()

            # Accumulate frequency maps
            if scene_freq_image is None:
                scene_freq_image = img_fft_centered_mod
            else:
                scene_freq_image = scene_freq_image + img_fft_centered_mod

            # Determine max resolution scale for this image
            e_total = img_fft_centered_mod.sum().item()
            e_min = e_total / self.start_significance_factor
            self.max_reso_scale = min(self.max_reso_scale, self._scale_solver(img_fft_centered_mod, e_min))

        # Average frequency map across all images
        scene_freq_image /= len(images)

        # Compute resolution schedule based on frequency content
        modulation_func = math.log

        self.reso_scales = []
        self.reso_level_significance = []
        self.reso_level_begin = []

        E_total = scene_freq_image.sum().item()
        E_min = self._compute_win_significance(scene_freq_image, self.max_reso_scale)

        # First level (lowest resolution)
        self.reso_level_significance.append(E_min)
        self.reso_scales.append(self.max_reso_scale)
        self.reso_level_begin.append(0)

        # Intermediate levels
        for i in range(1, self.reso_sample_num - 1):
            significance = (E_total - E_min) * i / (self.reso_sample_num - 1) + E_min
            self.reso_level_significance.append(significance)
            self.reso_scales.append(self._scale_solver(scene_freq_image, significance))

            # Compute when this level should begin (in iterations)
            self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
            self.reso_level_begin.append(
                int(self.increase_reso_until * self.reso_level_significance[-2] / modulation_func(E_total / E_min))
            )

        # Final level (full resolution)
        self.reso_level_significance.append(modulation_func(E_total / E_min))
        self.reso_scales.append(1.)
        self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
        self.reso_level_begin.append(
            int(self.increase_reso_until * self.reso_level_significance[-2] / modulation_func(E_total / E_min))
        )
        self.reso_level_begin.append(self.increase_reso_until)

        print(f"[ INFO ] FFT scheduler initialized with {len(self.reso_scales)} resolution levels")
        print(f"[ INFO ] Max resolution scale: {self.max_reso_scale:.2f}")

    def _init_simple_scheduler(self):
        """Use paper's resolution progression: 1/5, 1/4, 1/3, 1/2, 1/1"""
        print("[ INFO ] Initializing paper-style resolution scheduler...")
        
        # Fix the increase_reso_until value
        if self.increase_reso_until <= 0:
            self.increase_reso_until = self.opt.iterations // 2
            print(f"[ INFO ] Fixed increase_reso_until to: {self.increase_reso_until}")
        
        # Paper's progression: 1/5, 1/4, 1/3, 1/2, 1/1 (larger minimum resolution)
        self.reso_scales = [5, 4, 3, 2, 1]  # Inverse of the scale factors
        num_stages = len(self.reso_scales)
        
        # Distribute stages evenly across training
        self.reso_level_begin = []
        for i in range(num_stages):
            begin_iter = int(self.increase_reso_until * i / num_stages)
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
        
        if iteration < self.reso_level_begin[1]:
            return 1.0 / self.reso_scales[0]
        
        # Update next_i to current position
        while (self.next_i < len(self.reso_level_begin) and 
               iteration >= self.reso_level_begin[self.next_i]):
            self.next_i += 1

        if self.next_i >= len(self.reso_level_begin):
            return 1.0  # Full resolution

        # Interpolate between resolution levels (DashGaussian style)
        i = self.next_i - 1
        i_now, i_nxt = self.reso_level_begin[i: i + 2]
        s_lst, s_now = self.reso_scales[i - 1: i + 1]

        # Smooth interpolation in frequency space
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

        # Debug output
        if self.current_iteration == 0:
            print(f"[ DEBUG ] Downsampled: {downsampled_height}x{downsampled_width}, "
                f"Tiles: {downsampled_height//self.tile_height}x{downsampled_width//self.tile_width}, "
                f"Scale: {scale:.3f}")

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
        
        # Scale focal lengths to maintain field of view
        downsampled_proj[0, 0] = proj_matrix[0, 0] / scale  # focal_x
        downsampled_proj[1, 1] = proj_matrix[1, 1] / scale  # focal_y
        
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