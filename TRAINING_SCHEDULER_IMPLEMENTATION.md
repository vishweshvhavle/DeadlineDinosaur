# Training Scheduler Implementation

This document describes the new `TrainingScheduler` class that implements momentum-based primitive budgeting and iteration-based frequency resolution scheduling.

## Overview

The `TrainingScheduler` replaces the time-based resolution scheduler with an **iteration-based** approach that:
- Uses **FFT analysis** to determine optimal resolution scales based on image frequency content
- Implements **momentum-based primitive budgeting** to control Gaussian densification
- Provides **smooth interpolation** between resolution levels
- Tracks densification rates based on current resolution and iteration

## Key Features

### 1. Momentum-Based Primitive Budgeting

The scheduler maintains a momentum value that smoothly tracks the densification rate:

```python
# Auto mode (default)
momentum = 5 * init_n_gaussian
max_n_gaussian = init_n_gaussian + momentum

# Fixed budget mode
max_n_gaussian = pipe.max_n_gaussian  # Set explicitly
```

The momentum is updated after each densification step:
```python
scheduler.update_momentum(num_gaussians_added)
```

### 2. Frequency-Based Resolution Scheduling

Instead of using fixed time intervals (1/6, 2/6, 3/6, etc.), the scheduler:
1. Analyzes training images using FFT to determine frequency content
2. Creates resolution levels based on frequency significance
3. Schedules resolution increases based on **iterations**, not time
4. Smoothly interpolates between resolution levels

### 3. Iteration-Based Progression

All scheduling is based on training iterations:
```python
# Get current resolution scale
scale = scheduler.get_res_scale(iteration)  # e.g., 8 for 1/8 resolution

# Get inverse scale (for compatibility)
inv_scale = scheduler.get_resolution_scale_inverse(iteration)  # e.g., 0.125

# Get densification rate
rate = scheduler.get_densify_rate(iteration, current_n_gaussians, scale)
```

## Usage

### Basic Setup

```python
from deadlinedino.training.scheduling_utils import TrainingScheduler
from deadlinedino.arguments import OptimizationParams, PipelineParams

# Initialize parameters
opt = OptimizationParams.get_class_default_obj()
pipe = PipelineParams.get_class_default_obj()

# Create scheduler
scheduler = TrainingScheduler(opt, pipe, gaussians, original_images)
```

### During Training Loop

```python
for iteration in range(max_iterations):
    # Get current resolution scale
    res_scale = scheduler.get_res_scale(iteration)

    # Downsample images based on scale
    target_height = original_height // res_scale
    target_width = original_width // res_scale

    # ... training code ...

    # During densification
    if iteration % densification_interval == 0:
        cur_n_gaussians = len(gaussians)
        max_densify_rate = scheduler.get_densify_rate(
            iteration, cur_n_gaussians, res_scale
        )

        # Perform densification with rate limit
        added = densify_gaussians(max_rate=max_densify_rate)

        # Update momentum
        scheduler.update_momentum(added)
```

## Configuration Parameters

### PipelineParams

```python
pipe.densify_mode = 'freq'  # 'freq' or 'free'
  # 'freq': Use frequency-based densification with momentum
  # 'free': No densification limits

pipe.resolution_mode = 'freq'  # 'freq' or 'const'
  # 'freq': Use FFT-based resolution scheduling
  # 'const': Use constant full resolution

pipe.max_n_gaussian = -1  # Maximum number of Gaussians
  # -1: Auto mode with momentum-based budgeting
  # >0: Fixed maximum budget
```

### OptimizationParams

```python
opt.iterations = 30000  # Total training iterations
opt.densify_until_iter = -1  # When to stop densification (-1 = auto = iterations//2)
opt.densification_interval = 100  # How often to densify
opt.position_lr_delay_mult = 0.01  # LR delay multiplier
```

## Differences from Time-Based Scheduler

| Aspect | Old (Time-Based) | New (Iteration-Based) |
|--------|------------------|----------------------|
| **Progression** | Based on elapsed seconds | Based on training iterations |
| **Resolution Levels** | Fixed 6 stages (1/6, 2/6, ..., 6/6) | Dynamic levels from FFT analysis |
| **Primitive Budget** | Not controlled | Momentum-based budgeting |
| **Densification** | Not scheduled | Rate-controlled based on resolution |
| **Interpolation** | Step-wise | Smooth frequency-space interpolation |
| **Reproducibility** | Depends on hardware speed | Deterministic based on iterations |

## Methods Reference

### `get_res_scale(iteration)`
Returns the resolution scale factor for the given iteration.
- **Returns**: Integer scale (e.g., 8 for 1/8 resolution, 1 for full resolution)

### `get_resolution_scale_inverse(iteration)`
Returns the inverse of the resolution scale (for compatibility).
- **Returns**: Float (e.g., 0.125 for 1/8 resolution, 1.0 for full resolution)

### `get_densify_rate(iteration, cur_n_gaussian, cur_scale)`
Returns the maximum densification rate for the current iteration.
- **iteration**: Current training iteration
- **cur_n_gaussian**: Current number of Gaussian primitives
- **cur_scale**: Current resolution scale
- **Returns**: Maximum rate (e.g., 0.2 = can add up to 20% more primitives)

### `update_momentum(momentum_step)`
Updates the momentum based on the number of primitives added.
- **momentum_step**: Number of primitives added in the last densification

### `lr_decay_from_iter()`
Returns the iteration from which to start learning rate decay.
- **Returns**: Iteration number (when resolution scale drops below 2)

## Example: Switching from Old to New Scheduler

### Old Code (Time-Based)
```python
from deadlinedino.training.scheduling_utils import ResolutionScheduler

# Initialize
resolution_scheduler = ResolutionScheduler(num_stages=6, stage_duration=9.0)
resolution_scheduler.start()

# In training loop
scale = resolution_scheduler.get_resolution_scale()
```

### New Code (Iteration-Based)
```python
from deadlinedino.training.scheduling_utils import TrainingScheduler

# Initialize
scheduler = TrainingScheduler(opt, pipe, gaussians, original_images)

# In training loop
scale = scheduler.get_resolution_scale_inverse(iteration)
```

## Implementation Details

The scheduler uses:
1. **FFT Analysis**: Computes 2D FFT of images to analyze frequency content
2. **Binary Search**: Finds optimal scales using `scale_solver()` function
3. **Log-Linear Interpolation**: Smooth transitions between resolution levels
4. **Momentum Integration**: Exponentially weighted moving average for budgeting

### Resolution Interpolation Formula

Between two resolution levels, the scale is interpolated using:

```
scale = sqrt(1 / (α * (1/s_now² - 1/s_last²) + 1/s_last²))
```

where:
- `α` = progress between current and next level (0 to 1)
- `s_last` = previous resolution scale
- `s_now` = current resolution scale

This provides smooth transitions in frequency space.

### Momentum Update Formula

```
momentum = max(momentum, integrate_factor * momentum + min(cap, step))
```

where:
- `integrate_factor` = 0.98 (decay factor)
- `cap` = 1,000,000 (maximum step size)
- `step` = number of primitives added

## Testing

A test script is provided: `test_training_scheduler.py`

Run with:
```bash
python test_training_scheduler.py
```

This tests:
- Resolution scale progression across iterations
- Momentum-based primitive budgeting
- Fixed budget mode
- FFT-based initialization

## Performance Characteristics

- **Initialization**: O(N * H * W * log(H*W)) where N = number of images
- **get_res_scale()**: O(1) amortized (with iterator tracking)
- **get_densify_rate()**: O(1)
- **update_momentum()**: O(1)

## Notes

1. The scheduler assumes images are on GPU for faster FFT computation
2. Images can be either torch.Tensor or numpy arrays (auto-converted)
3. The `next_i` iterator is maintained for efficient resolution lookups
4. All parameters have sensible defaults for auto mode
5. The scheduler is compatible with the existing GaussianModel interface
