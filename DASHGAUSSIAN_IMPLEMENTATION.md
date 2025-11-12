# DashGaussian Progressive Fitting Strategy Implementation

This document describes the implementation of the DashGaussian progressive fitting strategy in DeadlineDinosaur.

## Overview

DashGaussian is a scheduling scheme that accelerates 3D Gaussian Splatting (3DGS) optimization by:
1. **Progressive Resolution Training**: Starting with low-resolution images and gradually increasing to full resolution
2. **Synchronized Primitive Growth**: Growing Gaussian primitives in sync with resolution to avoid wasteful densification
3. **Delayed Learning Rate Decay**: Maintaining high XYZ learning rate until maximum resolution is reached

## Key Components

### 1. Training Scheduler (`deadlinedino/training/schedule_utils.py`)

New file implementing the `TrainingScheduler` class with:

- **Resolution Scheduling**: Frequency-based progression from 1/8 resolution to full resolution
- **Primitive Budget Control**: Momentum-based tracking of primitive growth
- **Densification Rate Calculation**: Synchronizes primitive addition with current resolution

Key parameters:
- `max_reso_scale`: 8 (start at 1/8 resolution)
- `reso_sample_num`: 32 (samples for significance calculation)
- `max_densify_rate_per_step`: 0.2 (maximum primitives to add per step)
- `integrate_factor`: 0.98 (momentum smoothing)

### 2. Modified Trainer (`deadlinedino/training/trainer.py`)

Changes:
- Import and initialize `TrainingScheduler` with training images
- Dynamically update resolution scale per iteration
- Resize ground truth images based on current resolution scale
- Pass scheduler information to density controller
- Update optimizer scheduler when max resolution is reached

Key additions:
```python
# Initialize scheduler (after line 97)
training_scheduler = TrainingScheduler(...)
current_render_scale = training_scheduler.max_reso_scale

# Update resolution per iteration (lines 147-164)
current_render_scale = training_scheduler.get_res_scale(current_iteration)
if current_render_scale > 1:
    gt_image = resize_image_with_scale(gt_image, current_render_scale)

# Update LR scheduler when max resolution reached (lines 151-155)
if training_scheduler.max_resolution_reached:
    schedular.decay_from_iter = training_scheduler.max_resolution_iter
```

### 3. Modified Density Controller (`deadlinedino/training/densify.py`)

Changes:
- Added `densify_rate` parameter to `split_and_clone()` methods
- Implemented progressive densification limiting based on rate
- TamingGS version uses densify_rate when provided, falls back to original logic otherwise

Key changes:
```python
def split_and_clone(self, optimizer, epoch, densify_rate=None):
    # Calculate budget based on densify_rate
    if densify_rate is not None:
        budget = int(densify_rate * self.init_points_num) + prune_num
    # Limit primitive addition accordingly
```

### 4. Modified Optimizer Scheduler (`deadlinedino/training/optimizer.py`)

Changes:
- Added `decay_from_iter` parameter to `Scheduler` class
- XYZ learning rate maintained at initial value until `decay_from_iter`
- After `decay_from_iter`, exponential decay proceeds as normal

Key changes:
```python
def __init__(self, ..., decay_from_iter=0):
    self.decay_from_iter = decay_from_iter

def __helper(self):
    if self.last_epoch < self.decay_from_iter:
        return self.lr_init  # Maintain initial LR
    # Otherwise apply exponential decay
```

## How It Works

### Resolution Progression

1. **Initialization**: Analyzes training images to calculate frequency significance
2. **Schedule Creation**: Maps iterations to resolution scales (8→4→2→1)
3. **Dynamic Rendering**: Downscales ground truth images during training
4. **Reaches Full Resolution**: Typically at 70% of total training iterations

### Primitive Growth Synchronization

1. **Calculate Target**: Based on current resolution scale
   - Lower resolution = fewer primitives needed
   - Growth proportional to (max_scale / current_scale)²

2. **Momentum Tracking**: Exponentially smoothed growth target
   - `momentum = 0.98 * momentum + 0.02 * target_add`

3. **Densify Rate**: Converts momentum to rate relative to initial primitives
   - Clamped to maximum of 0.2 per step

4. **Budget Limiting**: Density controller uses rate to limit clone/split operations

### Delayed XYZ Learning Rate Decay

1. **Initial Phase**: XYZ LR maintained at `position_lr_init`
2. **Max Resolution Detected**: When resolution scale reaches 1.0
3. **Decay Activation**: Sets `decay_from_iter` in optimizer scheduler
4. **Progressive Decay**: XYZ LR decays exponentially from max resolution point onward

## Benefits for 60-Second Constraint

1. **Faster Early Iterations**: Low resolution rendering is faster
2. **Efficient Primitive Growth**: Avoids over-densification at low resolutions
3. **Better Structure Formation**: Coarse-to-fine approach captures structure first
4. **Optimized LR Schedule**: Maintains geometric flexibility longer

## Configuration

The progressive strategy is **automatically enabled** when training. No additional parameters required.

To adjust behavior, modify parameters in `schedule_utils.py`:
- `max_reso_scale`: Change starting resolution (default: 8)
- `max_densify_rate_per_step`: Control primitive growth speed (default: 0.2)
- `integrate_factor`: Adjust momentum smoothing (default: 0.98)

## Monitoring

During training, watch for:
```
[DashGaussian] Max resolution reached at iteration XXXX
[DashGaussian] XYZ learning rate decay will start from this point
```

This indicates the transition point where:
- Resolution reaches 1.0 (full size)
- Primitive growth becomes less aggressive
- XYZ learning rate begins exponential decay

## Compatibility

The implementation:
- ✅ Works with existing TamingGS densification
- ✅ Compatible with clustering mode (`cluster_size > 0`)
- ✅ Compatible with sparse gradients
- ✅ Compatible with learnable viewproj
- ✅ Falls back gracefully if scheduler not provided

## Technical Details

### Image Resizing
- Uses PyTorch's bilinear interpolation with antialiasing
- Function: `resize_image_with_scale(image_tensor, scale)`
- Equivalent to Lanczos-like filtering

### Frequency-Based Schedule
- Analyzes image gradients at multiple scales
- Uses Sobel filters to detect high-frequency content
- Schedules resolution increase based on cumulative significance

### Primitive Budget Formula
```python
resolution_factor = (max_reso_scale / render_scale) ** 2
target_at_resolution = init_n + (max_n - init_n) * (resolution_factor / max_reso_scale²)
momentum = integrate_factor * momentum + (1 - integrate_factor) * target_add
densify_rate = momentum / init_n
```

## Files Modified

1. `deadlinedino/training/schedule_utils.py` - **NEW FILE**
2. `deadlinedino/training/trainer.py` - Modified
3. `deadlinedino/training/densify.py` - Modified
4. `deadlinedino/training/optimizer.py` - Modified

## References

- DashGaussian Paper: Progressive 3D Gaussian Splatting with resolution scheduling
- TamingGS: Importance-based primitive growth control
- Original 3DGS: Base Gaussian splatting framework
