# Bug Fix: Black Rendering Issue During Progressive Resolution Training

## Problem

During progressive resolution training (DashGaussian), the rendered images would become completely black (mean=0.0) around iteration 1500-2000, causing training loss to become NaN and preventing successful training.

## Root Cause

The issue was caused by **in-place modification of shared projection matrix tensors** in the training loop.

### Detailed Explanation

1. **Dataset Preloading**: When `pp.device_preload=True`, the dataset preloads camera projection matrices as CUDA tensors and stores them (see `data.py:184-189`). These tensors are shared across all iterations.

2. **In-Place Modification**: During training with progressive resolution scaling, the code modified the projection matrix focal lengths in-place:
   ```python
   # OLD BUGGY CODE (trainer.py:177-178)
   proj_matrix[:, 0, 0] = proj_matrix[:, 0, 0] / current_render_scale
   proj_matrix[:, 1, 1] = proj_matrix[:, 1, 1] / current_render_scale
   ```

3. **Accumulating Error**: Since the training dataset is shuffled and views are reused across multiple epochs, the same projection matrix tensor would be:
   - Loaded from the dataset (with already-modified values)
   - Divided by `current_render_scale` again
   - This accumulated over many iterations

4. **Catastrophic Failure**: After enough iterations, the projection matrix values became so small (divided by scale 10+ times) that:
   - The rendering projection became incorrect
   - Rendered images became completely black
   - Training loss became NaN

### Example

If `current_render_scale = 2` and a view is processed 4 times:
- Iteration 1: `proj_matrix[0,0] = 2.0 / 2 = 1.0`
- Iteration 2: `proj_matrix[0,0] = 1.0 / 2 = 0.5`  ❌ Should be 1.0!
- Iteration 3: `proj_matrix[0,0] = 0.5 / 2 = 0.25` ❌ Should be 1.0!
- Iteration 4: `proj_matrix[0,0] = 0.25 / 2 = 0.125` ❌ Should be 1.0!

After just 4 iterations, the focal length is 16x smaller than it should be!

## Solution

**Create a copy (clone) of the projection matrix before modifying it**, ensuring that the shared dataset tensor remains unchanged for future iterations.

### Changes Made

#### 1. Progressive Resolution Scaling (trainer.py:176-183)
```python
# Apply progressive resolution scaling
if current_render_scale > 1:
    gt_image = resize_image_with_scale(gt_image, current_render_scale)
    # CRITICAL: Create a copy of projection matrix to avoid in-place modification
    # of the shared tensor from the dataset. Otherwise, the projection matrix
    # gets divided multiple times as views are reused across epochs.
    proj_matrix = proj_matrix.clone()
    # Adjust projection matrix focal lengths for the new resolution
    proj_matrix[:, 0, 0] = proj_matrix[:, 0, 0] / current_render_scale
    proj_matrix[:, 1, 1] = proj_matrix[:, 1, 1] / current_render_scale
```

#### 2. Learnable ViewProj (trainer.py:185-189)
```python
if op.learnable_viewproj:
    # Create copies to avoid modifying shared dataset tensors
    view_matrix = view_matrix.clone()
    if current_render_scale == 1:  # Already cloned above if > 1
        proj_matrix = proj_matrix.clone()
    # ... rest of learnable viewproj code
```

#### 3. Evaluation Loop (trainer.py:263-266)
```python
if name=="Trainingset" and op.learnable_viewproj:
    # Create copies to avoid modifying shared dataset tensors
    view_matrix = view_matrix.clone()
    proj_matrix = proj_matrix.clone()
    # ... rest of evaluation code
```

## Impact

- ✅ Prevents black rendering during progressive resolution training
- ✅ Fixes NaN loss issues after iteration 1500+
- ✅ Enables stable training through all resolution scales
- ✅ Minimal performance overhead (cloning small 4x4 matrices once per iteration)

## Testing

To verify the fix works:
1. Run training with progressive resolution on any scene
2. Monitor the rendered image statistics at iterations 1500, 2000, 2500, etc.
3. Images should maintain reasonable mean values (not 0.0)
4. Loss should remain finite (not NaN)

## Files Modified

- `deadlinedino/training/trainer.py`: Added `.clone()` calls before modifying projection and view matrices
- `test_projection_matrix_fix.py`: Created test demonstrating the bug and fix

## Related Code

- Dataset preloading: `data.py:178-198` (CameraFrameDataset.__init__)
- Progressive resolution: `deadlinedino/training/schedule_utils.py`
- Rendering: `deadlinedino/render/__init__.py:34-98`
