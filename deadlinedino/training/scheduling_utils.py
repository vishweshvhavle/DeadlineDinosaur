import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import time


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
    Time-based resolution scheduler that progressively increases resolution during training.

    The resolution increases in discrete steps based on elapsed time:
    - 0-9s: 1/6 resolution
    - 9-18s: 2/6 resolution
    - 18-27s: 3/6 resolution
    - 27-36s: 4/6 resolution
    - 36-45s: 5/6 resolution
    - 45s+: 6/6 (full) resolution
    """

    def __init__(self, num_stages: int = 6, stage_duration: float = 9.0):
        """
        Initialize the resolution scheduler.

        Args:
            num_stages: Number of resolution stages (default: 6)
            stage_duration: Duration of each stage in seconds (default: 9.0)
        """
        self.num_stages = num_stages
        self.stage_duration = stage_duration
        self.start_time = None

    def start(self):
        """Start the timer for the resolution scheduler."""
        self.start_time = time.time()

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

        Returns:
            Resolution scale (e.g., 1/6, 2/6, ..., 6/6)
        """
        stage = self.get_current_stage()
        return stage / self.num_stages

    def get_downsampled_shape(self, full_height: int, full_width: int) -> tuple[int, int]:
        """
        Get the downsampled image shape based on current resolution scale.

        Args:
            full_height: Full resolution height
            full_width: Full resolution width

        Returns:
            (height, width) tuple for current resolution
        """
        scale = self.get_resolution_scale()
        downsampled_height = int(full_height * scale)
        downsampled_width = int(full_width * scale)
        # Ensure at least 1 pixel
        downsampled_height = max(1, downsampled_height)
        downsampled_width = max(1, downsampled_width)
        return downsampled_height, downsampled_width

    def get_downsampled_proj_matrix(self, proj_matrix: np.ndarray,
                                     full_height: int, full_width: int) -> np.ndarray:
        """
        Get a projection matrix adjusted for center crop rendering at reduced resolution.

        For center crop: we keep the same field of view but render at smaller resolution.
        This requires dividing the focal lengths by the scale factor.

        Example: 100x100 image reduced by half (scale=0.5) -> 50x50 center region
        - Same angular FOV maintained
        - Rendered image is 50x50 (center crop of what would be 100x100)
        - GT is downsampled to 50x50 to match

        Args:
            proj_matrix: Original projection matrix (4x4)
            full_height: Full resolution height
            full_width: Full resolution width

        Returns:
            Adjusted projection matrix for center crop at current resolution
        """
        scale = self.get_resolution_scale()

        # Create a copy of the projection matrix
        downsampled_proj = proj_matrix.copy()

        # For center crop: divide focal lengths by scale to maintain same FOV at smaller resolution
        # proj_matrix[0,0] is focal_x, proj_matrix[1,1] is focal_y
        # This creates a center crop effect: same FOV rendered at reduced resolution
        downsampled_proj[0, 0] = proj_matrix[0, 0] / scale
        downsampled_proj[1, 1] = proj_matrix[1, 1] / scale

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
