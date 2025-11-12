import torch
import numpy as np
import cv2


class TrainingScheduler:
    """
    DashGaussian-style progressive training scheduler.

    Implements:
    - Progressive resolution training (low to high frequency)
    - Synchronized primitive growth with resolution
    - Momentum-based primitive budget control
    """

    def __init__(self, opt, pipe, init_n_gaussian, original_images, total_iterations=30000):
        """
        Initialize the progressive training scheduler.

        Args:
            opt: Optimization parameters
            pipe: Pipeline parameters
            init_n_gaussian: Initial number of Gaussian primitives
            original_images: List of original training images for significance calculation
            total_iterations: Total training iterations
        """
        self.opt = opt
        self.pipe = pipe
        self.total_iterations = total_iterations

        # Resolution scheduling parameters
        self.resolution_mode = "freq"  # frequency-based progression
        self.max_reso_scale = 8  # Start at 1/8 resolution
        self.reso_sample_num = 32  # Number of sample points for significance
        self.current_reso_scale = self.max_reso_scale  # Start with lowest resolution

        # Primitive scheduling parameters
        self.init_n_gaussian = init_n_gaussian
        self.max_n_gaussian = pipe.target_primitives if hasattr(pipe, 'target_primitives') and pipe.target_primitives else (init_n_gaussian * 6)

        # Momentum tracking for primitive growth
        self.momentum = -1 if (hasattr(pipe, 'target_primitives') and pipe.target_primitives) else 5 * init_n_gaussian
        self.integrate_factor = 0.98  # Smoothing factor for momentum

        # Max densify rate per step
        self.max_densify_rate_per_step = 0.2

        # Calculate resolution schedule based on image significance
        self.init_reso_scheduler(original_images)

        # Track when we reach max resolution (for delayed LR decay)
        self.max_resolution_reached = False
        self.max_resolution_iter = None

    def init_reso_scheduler(self, original_images):
        """
        Initialize resolution scheduler based on image frequency significance.

        Analyzes image gradients to determine frequency content and schedules
        resolution progression from low to high frequencies.

        Args:
            original_images: List of training images (torch.Tensor or numpy arrays)
        """
        if self.resolution_mode != "freq":
            # Simple linear progression
            self.significance_levels = None
            return

        # Calculate significance levels based on image gradients
        significance_scores = []

        for img in original_images:
            if isinstance(img, torch.Tensor):
                img_np = img.cpu().numpy()
            else:
                img_np = img

            # Convert to grayscale if needed
            if img_np.ndim == 3:
                if img_np.shape[0] == 3:  # CHW format
                    img_np = img_np.transpose(1, 2, 0)
                img_gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_np.astype(np.uint8)

            # Calculate gradient magnitude at different scales
            scale_significance = []
            for scale in range(1, int(np.log2(self.max_reso_scale)) + 1):
                downscale_factor = 2 ** scale
                h, w = img_gray.shape
                small_img = cv2.resize(img_gray, (w // downscale_factor, h // downscale_factor))

                # Calculate gradients
                grad_x = cv2.Sobel(small_img, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(small_img, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)

                # Use high percentile as significance metric
                significance = np.percentile(grad_mag, 95)
                scale_significance.append(significance)

            significance_scores.append(scale_significance)

        # Average across all images
        avg_significance = np.mean(significance_scores, axis=0)

        # Normalize and create cumulative distribution
        self.significance_levels = avg_significance / avg_significance.sum()
        self.cumulative_significance = np.cumsum(self.significance_levels)

        # Define resolution schedule milestones
        # Map iteration progress to resolution scale
        self.resolution_schedule = self._create_resolution_schedule()

    def _create_resolution_schedule(self):
        """
        Create a schedule mapping iteration -> resolution scale.

        Returns:
            List of (iteration, resolution_scale) tuples
        """
        if self.resolution_mode != "freq" or self.significance_levels is None:
            # Linear schedule: reach full resolution at 70% of training
            num_steps = int(np.log2(self.max_reso_scale)) + 1
            schedule = []
            for i, scale in enumerate([2**k for k in range(int(np.log2(self.max_reso_scale)), -1, -1)]):
                iter_point = int((i / num_steps) * 0.7 * self.total_iterations)
                schedule.append((iter_point, scale))
            return schedule

        # Frequency-based schedule
        schedule = []
        start_significance_factor = 4  # Start training when significance is low

        for i, scale in enumerate([2**k for k in range(int(np.log2(self.max_reso_scale)), -1, -1)]):
            # Calculate when this resolution should be activated
            # based on cumulative significance
            if i < len(self.cumulative_significance):
                progress = self.cumulative_significance[i] / start_significance_factor
            else:
                progress = 1.0

            progress = min(progress, 0.7)  # Reach full resolution by 70% of training
            iter_point = int(progress * self.total_iterations)
            schedule.append((iter_point, scale))

        return schedule

    def get_res_scale(self, iteration):
        """
        Get the current resolution scale for this iteration.

        Args:
            iteration: Current training iteration

        Returns:
            float: Resolution scale factor (1 = full resolution, 2 = half, etc.)
        """
        # Find the appropriate resolution scale
        current_scale = self.max_reso_scale

        for iter_threshold, scale in self.resolution_schedule:
            if iteration >= iter_threshold:
                current_scale = scale
            else:
                break

        # Update internal state
        old_scale = self.current_reso_scale
        self.current_reso_scale = current_scale

        # Track when we reach maximum resolution
        if current_scale == 1 and not self.max_resolution_reached:
            self.max_resolution_reached = True
            self.max_resolution_iter = iteration

        return current_scale

    def get_densify_rate(self, iteration, current_n_primitives, render_scale):
        """
        Calculate the densification rate synchronized with resolution.

        Primitives should grow proportionally with resolution to avoid
        wasteful densification at low resolutions.

        Args:
            iteration: Current training iteration
            current_n_primitives: Current number of primitives
            render_scale: Current rendering resolution scale

        Returns:
            float: Densification rate (fraction of init_n_gaussian to add)
        """
        # Calculate target primitive count at current resolution
        # Lower resolution needs fewer primitives
        resolution_factor = (self.max_reso_scale / render_scale) ** 2
        target_at_resolution = self.init_n_gaussian + (self.max_n_gaussian - self.init_n_gaussian) * (resolution_factor / (self.max_reso_scale ** 2))

        # Calculate how many primitives we need to add
        target_add = target_at_resolution - current_n_primitives

        # Update momentum (exponential moving average)
        if self.momentum < 0:
            # First time: initialize
            self.momentum = max(target_add, 0)
        else:
            # Smooth update
            self.momentum = self.integrate_factor * self.momentum + (1 - self.integrate_factor) * max(target_add, 0)

        # Calculate densify rate based on momentum
        densify_rate = self.momentum / self.init_n_gaussian

        # Clamp to reasonable limits
        densify_rate = min(densify_rate, self.max_densify_rate_per_step)
        densify_rate = max(densify_rate, 0.0)

        return densify_rate

    def update_momentum(self, momentum_add):
        """
        Manually update momentum (called after densification).

        Args:
            momentum_add: Amount to add to momentum
        """
        self.momentum += momentum_add

    def should_start_xyz_decay(self, iteration):
        """
        Determine if XYZ learning rate decay should start.

        According to DashGaussian, XYZ LR decay should be delayed until
        maximum resolution is reached.

        Args:
            iteration: Current training iteration

        Returns:
            bool: Whether to start XYZ LR decay
        """
        if not self.max_resolution_reached:
            return False

        return iteration >= self.max_resolution_iter

    def get_lr_decay_iter_offset(self):
        """
        Get the iteration offset for LR decay (delay decay until max resolution).

        Returns:
            int: Number of iterations to delay decay
        """
        if self.max_resolution_iter is None:
            return 0
        return self.max_resolution_iter


def resize_image_with_scale(image_tensor, scale):
    """
    Resize image tensor with Lanczos-like interpolation.

    Args:
        image_tensor: Input image tensor (C, H, W) or (B, C, H, W)
        scale: Downscale factor (1 = no change, 2 = half size, etc.)

    Returns:
        torch.Tensor: Resized image
    """
    if scale == 1:
        return image_tensor

    # Handle both 3D and 4D tensors
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    h, w = image_tensor.shape[-2:]
    new_h, new_w = int(h / scale), int(w / scale)

    # Use bilinear with antialiasing (similar to Lanczos)
    resized = torch.nn.functional.interpolate(
        image_tensor,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False,
        antialias=True
    )

    if squeeze_output:
        resized = resized.squeeze(0)

    return resized
