import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import time
import math
import sys
import os

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
    Time-based resolution scheduler with smooth scaling and CUDA-compatible dimensions.
    Maintains aspect ratio while ensuring tile-aligned dimensions.
    """

    def __init__(self, opt, pipe, original_images: list = None, 
                 tile_height: int = 8, tile_width: int = 16):
        """
        Initialize the resolution scheduler.

        Args:
            opt: OptimizationParams with training parameters
            pipe: PipelineParams with pipeline configuration  
            original_images: List of training images (unused, kept for compatibility)
            tile_height: Tile height for CUDA kernels
            tile_width: Tile width for CUDA kernels
        """
        self.opt = opt
        self.pipe = pipe
        self.tile_height = tile_height
        self.tile_width = tile_width
        
        # Resolution parameters
        self.resolution_mode = getattr(pipe, 'resolution_mode', 'time')
        
        # Time-based schedule (seconds)
        self.start_time = None
        self.current_time = 0.0
        
        # Smooth scaling schedule: 
        self.scaling_duration = 6.0  # 6 seconds to go from 1/5 to full scale
        self.full_scale_duration = 54.0  # 54 seconds at full scale
        self.total_duration = 60.0

    def start_timing(self):
        """Start the timing for the scheduler."""
        self.start_time = time.time()
        self.current_time = 0.0

    def step(self):
        """Update the current time."""
        if self.start_time is not None:
            self.current_time = time.time() - self.start_time
        else:
            print("[ WARNING ] Resolution scheduler step called before start_timing()")

    def get_resolution_scale(self) -> float:
        """
        Get the current resolution scale factor based on elapsed time.
        Uses smooth interpolation from 1/5 to 1.0.
        """
        if self.resolution_mode == "const":
            return 1.0
        
        if self.start_time is None:
            return 0.2  # 1/5 scale
        
        if self.current_time >= self.scaling_duration:
            # After 40s, use full scale
            return 1.0
        
        # Smooth interpolation from 0.2 to 1.0 over 40 seconds
        # Using cosine interpolation for smoothness
        t = self.current_time / self.scaling_duration  # 0 to 1
        # Cosine interpolation: smoother than linear
        smooth_t = (1 - math.cos(t * math.pi)) / 2
        scale = 0.2 + (1.0 - 0.2) * smooth_t
        
        return scale

    def get_downsampled_dimensions(self, full_height: int, full_width: int) -> tuple[int, int]:
        """
        Get tile-aligned dimensions that maintain aspect ratio.
        Strategy: Scale both dimensions by the same factor, then round to tile multiples.
        """
        scale = self.get_resolution_scale()
        
        if scale >= 0.99:  # Close to full scale
            return full_height, full_width
        
        # Calculate target dimensions
        target_height = full_height * scale
        target_width = full_width * scale
        
        # Round to nearest tile multiple (maintains aspect ratio better than ceiling)
        downsampled_height = round(target_height / self.tile_height) * self.tile_height
        downsampled_width = round(target_width / self.tile_width) * self.tile_width
        
        # Ensure minimum dimensions (at least 2 tiles in each dimension)
        min_height = self.tile_height * 2
        min_width = self.tile_width * 2
        downsampled_height = max(min_height, downsampled_height)
        downsampled_width = max(min_width, downsampled_width)
        
        # Clamp to original dimensions
        downsampled_height = min(downsampled_height, full_height)
        downsampled_width = min(downsampled_width, full_width)
        
        return downsampled_height, downsampled_width

    def get_downsampled_proj_matrix(self, proj_matrix: np.ndarray,
                                     full_height: int, full_width: int) -> np.ndarray:
        """
        Adjust projection matrix for downsampled resolution.
        """
        ds_height, ds_width = self.get_downsampled_dimensions(full_height, full_width)
        
        # Calculate actual scale ratios used
        height_scale = ds_height / full_height
        width_scale = ds_width / full_width
        
        downsampled_proj = proj_matrix.copy()
        
        # Scale focal lengths and principal points
        downsampled_proj[0, 0] = proj_matrix[0, 0] * width_scale  # fx
        downsampled_proj[1, 1] = proj_matrix[1, 1] * height_scale  # fy
        downsampled_proj[0, 2] = proj_matrix[0, 2] * width_scale  # cx
        downsampled_proj[1, 2] = proj_matrix[1, 2] * height_scale  # cy
        
        return downsampled_proj

    @staticmethod
    def downsample_image_hq(image: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """
        High-quality image downsampling using torch interpolation.
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
        """
        scale = self.get_resolution_scale()

        return {
            'elapsed_time': self.current_time,
            'scale': scale,
            'current_scale': f"1/{1.0/scale:.2f}" if scale < 1.0 else "1.0",
            'timing_started': self.start_time is not None
        }