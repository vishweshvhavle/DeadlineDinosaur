# Debugging Tools for Gaussian Splatting Training

This document explains the debugging tools added to help diagnose rendering issues (e.g., black images, low PSNR).

## Quick Diagnosis Scripts

### 1. Inspect Point Cloud
Quickly check if your COLMAP point cloud has valid colors:

```bash
python inspect_pointcloud.py data/test/1748243104741
```

This will show:
- Number of points
- Position and color statistics
- Sample points with their colors
- Warnings if colors are too dark or missing

### 2. Test Rendering Components
Test each component of the pipeline individually:

```bash
python test_rendering_components.py data/test/1748243104741
```

This runs tests on:
- COLMAP data loading
- Gaussian parameter initialization
- Spherical harmonics to RGB conversion

## Debug Mode Training

The training code now includes comprehensive debugging. When you run training, debug information is automatically saved to `<output_dir>/debug/`.

### What Gets Logged

1. **Initial Point Cloud Statistics** (console output)
   - Number of points
   - XYZ range and statistics
   - Color range and statistics (should be [0, 1])
   - Warnings for NaN, Inf, or out-of-range values

2. **Initialized Gaussian Parameters** (console output)
   - Statistics for all parameters: XYZ, scale, rotation, SH coefficients, opacity
   - Activated values (after exp/sigmoid)
   - Warnings for very low opacity

3. **Training Iteration Logs** (saved to `debug/stats.jsonl`)
   - Loss values (total, L1, SSIM)
   - Gaussian statistics (opacity, scale)
   - Rendered image statistics (min, max, mean, std)
   - Ground truth image statistics

4. **Comparison Images** (saved to `debug/images/`)
   - Side-by-side rendered vs ground truth images
   - Saved every 100 iterations and for first 5 iterations
   - Format: `train_iter_XXXXX.png`

### Example Usage

```bash
python example_train.py \
  -s data/test/1748243104741 \
  -m outputs/debug_run \
  --sh_degree 3 \
  --source_type colmap \
  --iterations 500
```

Then check:
```bash
# View debug logs
cat outputs/debug_run/debug/stats.jsonl

# View comparison images
ls outputs/debug_run/debug/images/
```

## Common Issues and Diagnostics

### Issue: Black Rendered Images (PSNR < 10)

**Check 1: Point Cloud Colors**
```bash
python inspect_pointcloud.py <scene_path>
```
- If colors are nearly black (mean < 0.1), your COLMAP reconstruction may have failed
- Try re-running COLMAP with different parameters

**Check 2: Gaussian Initialization**
```bash
python test_rendering_components.py <scene_path>
```
Look for:
- SH_0 coefficients should NOT all be near zero
- Opacity mean should be around 0.1 (sigmoid of initialization)
- No NaN or Inf values

**Check 3: Training Dynamics**
Look at `debug/stats.jsonl`:
```bash
# Check if rendered images are black throughout training
cat debug/stats.jsonl | grep "rendered_image"
```
- If `rendered_image.mean` stays near 0.0, colors are not being learned
- If `opacity_mean` decreases to near 0.0, Gaussians are becoming transparent

**Check 4: View Rendered Images**
```bash
ls debug/images/
```
- Open the first few iterations (e.g., `train_iter_00000.png`)
- Left side = rendered, right side = ground truth
- If rendered side is completely black from iteration 0, initialization failed

### Issue: Low PSNR (10-20)

This could indicate:
- Resolution mismatch between training and evaluation
- Incorrect focal length in camera parameters
- Too few training iterations

Check:
1. Training logs for actual image resolution used
2. Progressive resolution schedule (DashGaussian)
3. Number of Gaussians vs target primitives

### Disabling Debug Mode

To disable debug output (for faster training), edit `deadlinedino/training/trainer.py`:

```python
# Line 33: Change enabled=True to enabled=False
debugger = debug_utils.init_debugger(lp.model_path, enabled=False)
```

## Understanding Debug Output

### stats.jsonl Format

Each line is a JSON object with:
```json
{
  "iteration": 100,
  "loss": 0.52,
  "l1_loss": 0.15,
  "ssim_loss": 0.37,
  "gaussians": {
    "count": 5000,
    "opacity_mean": 0.25,
    "opacity_min": 0.001,
    "opacity_max": 0.95,
    "scale_mean": 0.015,
    "scale_min": 0.0001,
    "scale_max": 0.5
  },
  "rendered_image": {
    "min": 0.0,
    "max": 0.98,
    "mean": 0.42,
    "std": 0.31
  },
  "gt_image": {
    "min": 0.0,
    "max": 1.0,
    "mean": 0.45
  }
}
```

### Key Metrics to Watch

1. **rendered_image.mean**: Should be > 0.2 for colored scenes
   - If < 0.05, images are too dark
   - Should be similar to gt_image.mean

2. **opacity_mean**: Should stay between 0.1-0.8
   - If approaching 0, Gaussians are disappearing
   - If approaching 1, might be overfitting

3. **loss**: Should decrease steadily
   - If not decreasing, learning rate may be too low
   - If NaN, learning rate may be too high

## Advanced Debugging

### Custom Checkpoints

Save specific iterations for inspection:
```python
python example_train.py ... --save_epochs 10 50 100 250 500
```

### Visualize Gaussian Parameters

The debug output includes all parameter statistics. You can plot these over time:

```python
import json
import matplotlib.pyplot as plt

# Load stats
stats = []
with open('outputs/debug_run/debug/stats.jsonl') as f:
    for line in f:
        stats.append(json.loads(line))

# Plot opacity over time
iterations = [s['iteration'] for s in stats]
opacity_mean = [s['gaussians']['opacity_mean'] for s in stats]

plt.plot(iterations, opacity_mean)
plt.xlabel('Iteration')
plt.ylabel('Mean Opacity')
plt.title('Gaussian Opacity During Training')
plt.show()
```

## Contact

If you've run all diagnostics and still have issues, please report with:
1. Output from `inspect_pointcloud.py`
2. Output from `test_rendering_components.py`
3. First 10 lines of `debug/stats.jsonl`
4. One sample comparison image from `debug/images/`
