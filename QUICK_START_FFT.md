# Quick Start Guide: FFT-Based ResolutionScheduler

This guide helps you get started with the enhanced ResolutionScheduler featuring FFT-based frequency analysis and high-quality downsampling.

## Setup

### 1. Build FastLanczos (Optional but Recommended)

FastLanczos provides high-quality Lanczos resampling. If not available, the system automatically falls back to PyTorch's area interpolation.

```bash
# Make the setup script executable
chmod +x setup_lanczos.sh

# Build FastLanczos
./setup_lanczos.sh

# Or manually:
cd deadlinedino/submodules/lanczos-resampling
python setup.py install --user
cd ../../..
```

### 2. Verify Installation

```bash
python -c "from deadlinedino.training.scheduling_utils import ResolutionScheduler; print('✓ ResolutionScheduler imported successfully')"
```

## Quick Examples

### Example 1: Basic Time-Based Scheduler (Existing Behavior)

```python
from deadlinedino.training.scheduling_utils import ResolutionScheduler

# Create scheduler
scheduler = ResolutionScheduler(num_stages=6, stage_duration=9.0)
scheduler.start()

# In your training loop
scale = scheduler.get_resolution_scale()
height, width = scheduler.get_downsampled_shape(1080, 1920)
```

### Example 2: FFT-Based Scheduler with Your Training Images

```python
from deadlinedino.training.scheduling_utils import ResolutionScheduler
from deadlinedino.data import CameraFrameDataset  # Your data loader

# Load your training images
images = []
for frame in training_frames[:50]:  # Use subset for analysis
    img = frame.load_image()  # Your image loading
    images.append(img)

# Create FFT-based scheduler
scheduler = ResolutionScheduler(
    num_stages=6,
    stage_duration=9.0,
    use_fft_analysis=True,  # Enable FFT
    images=images
)

scheduler.start()

# The scheduler now adapts to your scene's frequency content!
```

### Example 3: High-Quality Downsampling in Training

```python
# In your training loop, when downsampling GT images:

# OLD (simple bilinear):
# gt_downsampled = torch.nn.functional.interpolate(
#     gt_image, size=(height, width), mode='bilinear'
# )

# NEW (high-quality with automatic fallback):
gt_downsampled = ResolutionScheduler.downsample_image_hq(
    gt_image,
    target_height=height,
    target_width=width,
    use_lanczos=True  # Uses Lanczos if available, else area interpolation
)
```

## Run Examples

We provide comprehensive examples demonstrating all features:

```bash
# Run all examples (requires CUDA)
python example_fft_scheduler.py
```

This will demonstrate:
1. Basic time-based scheduling
2. FFT-based frequency analysis
3. High-quality downsampling (single and batch)
4. Training loop integration

## Integration with Your Trainer

The trainer (`deadlinedino/training/trainer.py`) has already been updated to use high-quality downsampling for debug visualizations.

To enable FFT-based scheduling in your training:

```python
# In your training script/config:

# Option 1: Keep existing time-based (default)
resolution_scheduler = ResolutionScheduler(
    num_stages=6,
    stage_duration=9.0
)

# Option 2: Enable FFT-based scheduling
# Load some training images first
sample_images = [...]  # Load subset of training images

resolution_scheduler = ResolutionScheduler(
    num_stages=6,
    stage_duration=9.0,
    use_fft_analysis=True,
    images=sample_images
)
```

## Key Benefits

### FFT-Based Scheduling
- **Adaptive**: Automatically adjusts resolution schedule based on scene complexity
- **Scene-aware**: High-frequency scenes maintain detail longer
- **Smooth transitions**: Interpolates between levels in frequency space

### High-Quality Downsampling
- **Lanczos resampling**: Superior quality when FastLanczos is available
- **Automatic fallback**: Uses PyTorch area interpolation if Lanczos unavailable
- **Batch support**: Efficiently processes batches of images
- **Better than bilinear**: Area interpolation preserves more detail than bilinear

## Performance Notes

### FFT Analysis
- One-time cost during initialization
- GPU-accelerated (fast)
- Recommended: Use 20-50 images for analysis (more = better estimate)

### Downsampling
- **Lanczos**: High quality but requires CPU-GPU transfers (~5-10ms overhead)
- **Area**: GPU-native, no transfers, faster (~1ms)
- **Recommendation**:
  - Training loop: Use `use_lanczos=False` for speed
  - Validation/Debug: Use `use_lanczos=True` for quality

## Troubleshooting

### FastLanczos not building?
```bash
# Make sure you have CUDA toolkit installed
nvcc --version

# Check PyTorch CUDA version matches
python -c "import torch; print(torch.version.cuda)"

# Try building with verbose output
cd deadlinedino/submodules/lanczos-resampling
python setup.py build_ext --verbose
```

### "FFT analysis requested but no images provided"
Make sure to pass a list of images when using `use_fft_analysis=True`:
```python
scheduler = ResolutionScheduler(
    use_fft_analysis=True,
    images=[img1, img2, img3]  # Don't forget this!
)
```

### Out of memory during FFT analysis
Reduce the number of images or use smaller resolutions:
```python
# Use fewer images
sample_images = training_images[::5]  # Every 5th image

# Or downsample before analysis
sample_images = [
    torch.nn.functional.interpolate(img, scale_factor=0.5)
    for img in training_images[:20]
]
```

## What's Different from DashGaussian?

Our implementation builds on DashGaussian's concepts with these enhancements:

1. **Standalone module**: Can be used independently
2. **Flexible downsampling**: Multiple methods with automatic fallback
3. **Backward compatible**: Existing code works without changes
4. **Better defaults**: Area interpolation vs bilinear
5. **Comprehensive testing**: Unit tests and examples included
6. **Documentation**: Extensive docs and usage examples

## Next Steps

1. ✓ Build FastLanczos for best quality
2. ✓ Run `example_fft_scheduler.py` to see it in action
3. ✓ Try FFT-based scheduling with your training data
4. ✓ Compare results with time-based scheduling
5. ✓ Tune `num_stages` and `stage_duration` for your use case

## Further Reading

- Full documentation: `RESOLUTION_SCHEDULER.md`
- Implementation details: `deadlinedino/training/scheduling_utils.py`
- Training integration: `deadlinedino/training/trainer.py`
- Reference: DashGaussian paper (frequency-aware training)

---

**Questions or issues?** Check `RESOLUTION_SCHEDULER.md` for detailed API documentation.
