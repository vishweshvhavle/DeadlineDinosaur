import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import time
import math
import sys
import os

# FastLanczos has been removed - using torch interpolation for all downsampling


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

    Uses FFT-based frequency analysis to determine optimal resolution scales based on image content.
    Supports both time-based progression and frequency-aware initialization.

    The resolution increases in discrete steps based on elapsed time:
    - 0-9s: 1/6 resolution
    - 9-18s: 2/6 resolution
    - 18-27s: 3/6 resolution
    - 27-36s: 4/6 resolution
    - 36-45s: 5/6 resolution
    - 45s+: 6/6 (full) resolution
    """

    def __init__(self, num_stages: int = 6, stage_duration: float = 9.0,
                 use_fft_analysis: bool = False, images: list = None):
        """
        Initialize the resolution scheduler.

        Args:
            num_stages: Number of resolution stages (default: 6)
            stage_duration: Duration of each stage in seconds (default: 9.0)
            use_fft_analysis: Whether to use FFT-based frequency analysis (default: False)
            images: List of training images for FFT analysis (required if use_fft_analysis=True)
        """
        self.num_stages = num_stages
        self.stage_duration = stage_duration
        self.start_time = None
        self.use_fft_analysis = use_fft_analysis

        # FFT-based parameters
        self.max_reso_scale = 8
        self.start_significance_factor = 4
        self.reso_sample_num = 32  # Must be no less than 2
        self.reso_scales = None
        self.reso_level_significance = None
        self.reso_level_begin = None
        self.next_i = 2

        # Initialize FFT-based scheduler if requested
        if use_fft_analysis:
            if images is None or len(images) == 0:
                print("[ WARNING ] FFT analysis requested but no images provided, falling back to time-based")
                self.use_fft_analysis = False
            else:
                self._init_fft_scheduler(images)

    def start(self):
        """Start the timer for the resolution scheduler."""
        self.start_time = time.time()

    def _compute_win_significance(self, significance_map: torch.Tensor, scale: float) -> float:
        """
        Compute the frequency significance within a centered window.

        Args:
            significance_map: FFT magnitude map (frequency domain)
            scale: Scale factor for window size

        Returns:
            Sum of frequency energy within the window
        """
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
        """
        Binary search to find the scale that captures target frequency significance.

        Args:
            significance_map: FFT magnitude map
            target_significance: Target frequency energy to capture

        Returns:
            Scale factor that captures the target significance
        """
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
        """
        Initialize resolution scales based on FFT analysis of training images.

        This method analyzes the frequency content of images to determine optimal
        resolution scales. Images with more high-frequency content will maintain
        higher resolutions for longer during training.

        Args:
            images: List of training images (torch.Tensor or numpy arrays)
        """
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
        total_duration = self.num_stages * self.stage_duration
        for i in range(1, self.reso_sample_num - 1):
            significance = (E_total - E_min) * i / (self.reso_sample_num - 1) + E_min
            self.reso_level_significance.append(significance)
            self.reso_scales.append(self._scale_solver(scene_freq_image, significance))

            # Compute when this level should begin (in seconds)
            self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
            self.reso_level_begin.append(
                int(total_duration * self.reso_level_significance[-2] / modulation_func(E_total / E_min))
            )

        # Final level (full resolution)
        self.reso_level_significance.append(modulation_func(E_total / E_min))
        self.reso_scales.append(1.)
        self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
        self.reso_level_begin.append(
            int(total_duration * self.reso_level_significance[-2] / modulation_func(E_total / E_min))
        )
        self.reso_level_begin.append(int(total_duration))

        print(f"[ INFO ] FFT scheduler initialized with {len(self.reso_scales)} resolution levels")
        print(f"[ INFO ] Max resolution scale: {self.max_reso_scale:.2f}")

    def get_current_stage(self) -> int:
        """
        Get the current resolution stage based on elapsed time.

        Returns:
            Current stage (1 to num_stages)
        """
        if self.start_time is None:
            return 1

        elapsed = time.time() - self.start_time
        stage = int(elapsed / self.stage_duration) + 1
        return min(stage, self.num_stages)

    def get_resolution_scale(self) -> float:
        """
        Get the current resolution scale factor.

        Uses FFT-based scales if initialized, otherwise uses time-based stages.

        Returns:
            Resolution scale (e.g., 1/6, 2/6, ..., 6/6)
        """
        if self.use_fft_analysis and self.reso_scales is not None:
            return self._get_fft_resolution_scale()
        else:
            stage = self.get_current_stage()
            return stage / self.num_stages

    def _get_fft_resolution_scale(self) -> float:
        """
        Get resolution scale based on FFT analysis and elapsed time.

        Returns:
            Resolution scale factor based on frequency content
        """
        if self.start_time is None:
            return 1.0 / self.reso_scales[0]

        elapsed = time.time() - self.start_time

        # Find current resolution level
        if elapsed < self.reso_level_begin[1]:
            return 1.0 / self.reso_scales[0]

        # Update next_i to current position
        while self.next_i < len(self.reso_level_begin) and elapsed >= self.reso_level_begin[self.next_i]:
            self.next_i += 1

        if self.next_i >= len(self.reso_level_begin):
            return 1.0  # Full resolution

        # Interpolate between resolution levels
        i = self.next_i - 1
        i_now, i_nxt = self.reso_level_begin[i: i + 2]
        s_lst, s_now = self.reso_scales[i - 1: i + 1]

        # Smooth interpolation in frequency space
        scale = (1 / ((elapsed - i_now) / (i_nxt - i_now) * (1/s_now**2 - 1/s_lst**2) + 1/s_lst**2))**0.5
        return 1.0 / scale

    def get_downsampled_shape(self, full_height: int, full_width: int,
                              tile_height: int = 8, tile_width: int = 16) -> tuple[int, int]:
        """
        Get the downsampled image shape based on current resolution scale.

        Ensures dimensions are compatible with CUDA tile-based rendering by:
        - Enforcing minimum dimensions (at least 2x tile size)
        - Rounding to multiples of tile size for better alignment

        Args:
            full_height: Full resolution height
            full_width: Full resolution width
            tile_height: Tile height for CUDA kernels (default: 8)
            tile_width: Tile width for CUDA kernels (default: 16)

        Returns:
            (height, width) tuple for current resolution
        """
        scale = self.get_resolution_scale()
        downsampled_height = int(full_height * scale)
        downsampled_width = int(full_width * scale)

        # Ensure minimum dimensions: at least 2x the tile size
        # This prevents CUDA kernel launch errors with very small resolutions
        min_height = tile_height * 2
        min_width = tile_width * 2
        downsampled_height = max(min_height, downsampled_height)
        downsampled_width = max(min_width, downsampled_width)

        # Round to nearest multiple of tile size for better alignment
        # This helps avoid edge cases in tile-based CUDA kernels
        downsampled_height = ((downsampled_height + tile_height - 1) // tile_height) * tile_height
        downsampled_width = ((downsampled_width + tile_width - 1) // tile_width) * tile_width

        return downsampled_height, downsampled_width

    def get_downsampled_proj_matrix(self, proj_matrix: np.ndarray,
                                     full_height: int, full_width: int) -> np.ndarray:
        """
        Get a projection matrix adjusted for rendering at reduced resolution.

        When downsampling, the focal lengths in the projection matrix must be INVERSELY
        scaled to maintain the same field of view at the reduced resolution.

        The projection matrix stores focal values as: focal_x = focal_pixels / (width * 0.5)
        When rendering at a lower resolution with scale < 1:
        - New resolution: width_new = width * scale, height_new = height * scale
        - To maintain the same view frustum (angular field of view), we need:
          focal_x_new = focal_pixels / (width_new * 0.5)
                      = focal_pixels / (width * scale * 0.5)
                      = (focal_pixels / (width * 0.5)) / scale
                      = focal_x_original / scale

        Example: 1000x1000 image reduced by scale=0.5 -> 500x500 image
        - Original focal_x in proj matrix = f / 500 (where f is focal length in pixels)
        - Downsampled focal_x = (f / 500) / 0.5 = f / 250 (focal value INCREASES)
        - This maintains the same angular FOV at the new 500x500 resolution

        Args:
            proj_matrix: Original projection matrix (4x4)
            full_height: Full resolution height
            full_width: Full resolution width

        Returns:
            Adjusted projection matrix for rendering at downsampled resolution
        """
    
        scale = self.get_resolution_scale()
        downsampled_proj = proj_matrix.copy()
        
        # For normalized projection matrices, NO scaling needed for focal values
        # The normalization already accounts for resolution changes
        
        # Only scale principal point (if non-zero and not centered)
        if abs(proj_matrix[0, 2]) > 1e-6:
            downsampled_proj[0, 2] = proj_matrix[0, 2] * scale
        if abs(proj_matrix[1, 2]) > 1e-6:
            downsampled_proj[1, 2] = proj_matrix[1, 2] * scale

        
        return downsampled_proj

    def get_info_dict(self) -> dict:
        """
        Get information about the current scheduler state.

        Returns:
            Dictionary with stage, scale, and elapsed time
        """
        stage = self.get_current_stage()
        scale = self.get_resolution_scale()
        elapsed = 0.0 if self.start_time is None else time.time() - self.start_time

        return {
            'stage': stage,
            'scale': scale,
            'elapsed_time': elapsed,
            'max_stages': self.num_stages
        }

    @staticmethod
    def downsample_image_hq(image: torch.Tensor, target_height: int, target_width: int,
                           use_lanczos: bool = True) -> torch.Tensor:
        """
        High-quality image downsampling using torch interpolation.

        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            target_height: Target height
            target_width: Target width
            use_lanczos: Unused parameter (kept for compatibility)

        Returns:
            Downsampled image tensor
        """
        if image.shape[-2] == target_height and image.shape[-1] == target_width:
            return image

        # Use torch interpolation
        return ResolutionScheduler._downsample_torch(image, target_height, target_width)

    @staticmethod
    def _downsample_torch(image: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """
        Downsample using torch.nn.functional.interpolate.

        Uses area interpolation for high-quality downsampling.

        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            target_height: Target height
            target_width: Target width

        Returns:
            Downsampled image tensor
        """
        original_shape = image.shape
        batch_mode = len(original_shape) == 4

        if not batch_mode:
            # Add batch dimension
            image = image.unsqueeze(0)

        # Use area interpolation for downsampling (better quality than bilinear)
        downsampled = torch.nn.functional.interpolate(
            image,
            size=(target_height, target_width),
            mode='area'
        )

        if not batch_mode:
            # Remove batch dimension
            downsampled = downsampled.squeeze(0)

        return downsampled
