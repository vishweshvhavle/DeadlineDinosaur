# Enhanced ResolutionScheduler with FFT-Based Analysis

## Overview

The `ResolutionScheduler` has been enhanced with FFT (Fast Fourier Transform) based frequency analysis and high-quality image downsampling capabilities. These improvements allow for intelligent resolution scheduling based on image content and superior image quality during progressive training.

## Key Features

### 1. FFT-Based Frequency Analysis

The scheduler can analyze the frequency content of training images to determine optimal resolution schedules. Images with more high-frequency detail will maintain higher resolutions for longer during training.

**How it works:**
- Performs FFT on training images to analyze frequency content
- Computes frequency energy distribution
- Determines resolution scales based on frequency significance
- Creates a smooth progression schedule that adapts to image complexity

### 2. High-Quality Downsampling

Supports multiple downsampling methods with automatic fallback:

- **FastLanczos**: High-quality Lanczos resampling (when available)
- **Torch Area Interpolation**: Better than bilinear for downsampling (fallback)

### 3. Backward Compatible

All existing code continues to work. FFT analysis is opt-in via parameters.

## Usage Examples

### Basic Usage (Time-Based, Existing Behavior)

```python
from training.scheduling_utils import ResolutionScheduler

# Create a time-based scheduler (existing behavior)
scheduler = ResolutionScheduler(num_stages=6, stage_duration=9.0)
scheduler.start()

# Get current resolution scale
scale = scheduler.get_resolution_scale()  # Returns 1/6, 2/6, ..., 6/6

# Get downsampled dimensions
height, width = scheduler.get_downsampled_shape(1080, 1920)

# Get adjusted projection matrix
proj_matrix_ds = scheduler.get_downsampled_proj_matrix(proj_matrix, height, width)
```

### Advanced Usage (FFT-Based)

```python
from training.scheduling_utils import ResolutionScheduler
import torch

# Load or generate training images
images = []
for image_path in training_image_paths:
    img = load_image(image_path)  # Your image loading function
    images.append(img)

# Create FFT-based scheduler
scheduler = ResolutionScheduler(
    num_stages=6,
    stage_duration=9.0,
    use_fft_analysis=True,  # Enable FFT analysis
    images=images           # Provide training images
)

scheduler.start()

# The scheduler will now use frequency-aware resolution scaling
scale = scheduler.get_resolution_scale()
```

### High-Quality Downsampling

```python
from training.scheduling_utils import ResolutionScheduler
import torch

# Single image downsampling
image = torch.rand((3, 1080, 1920))  # (C, H, W)
downsampled = ResolutionScheduler.downsample_image_hq(
    image,
    target_height=540,
    target_width=960,
    use_lanczos=True  # Use Lanczos if available, else fallback
)

# Batch downsampling
batch_images = torch.rand((8, 3, 1080, 1920))  # (B, C, H, W)
downsampled_batch = ResolutionScheduler.downsample_image_hq(
    batch_images,
    target_height=540,
    target_width=960,
    use_lanczos=True
)
```

## Implementation Details

### FFT Analysis Process

1. **Frequency Domain Transform**: Converts images to frequency domain using 2D FFT
2. **Magnitude Computation**: Computes magnitude of frequency components
3. **Window-Based Analysis**: Uses centered windows to measure frequency significance
4. **Scale Solver**: Binary search to find optimal scales for target frequency capture
5. **Schedule Generation**: Creates smooth progression from low to high resolution

### Resolution Scale Interpolation

When using FFT-based scheduling, the resolution scale is interpolated smoothly between discrete levels using the formula:

```
scale = 1 / sqrt(1 / ((t - t_now) / (t_next - t_now) * (1/s_now^2 - 1/s_last^2) + 1/s_last^2))
```

This provides smooth transitions in frequency space rather than abrupt jumps.

### Downsampling Methods

#### FastLanczos
- Uses Lanczos-2 kernel for high-quality resampling
- Operates on CPU (automatically handles GPU transfers)
- Preserves high-frequency details better than standard interpolation

#### Torch Area Interpolation (Fallback)
- Uses `mode='area'` which is better than bilinear for downsampling
- Averages pixels within each target pixel's area
- GPU-native, no data transfers needed

## Performance Considerations

### FFT Analysis
- Performed once during initialization
- GPU-accelerated for faster processing
- Negligible overhead after initialization

### Lanczos Downsampling
- Requires CPU-GPU transfers
- Higher quality but slightly slower than torch interpolation
- Best for debug/validation rather than training loop

### Recommended Usage
- **Training loop**: Use time-based scheduler (default) for speed
- **Validation/Debug**: Use FFT-based scheduler with Lanczos for quality
- **Production**: Enable FFT if image quality is critical and you have diverse scenes

## Integration with Trainer

The trainer has been updated to use high-quality downsampling for debug visualizations:

```python
# In trainer.py (already integrated)
downsampled_gt = resolution_scheduler.downsample_image_hq(
    debug_view_data['gt_image'],
    target_height=ds_height,
    target_width=ds_width,
    use_lanczos=True
)
```

## Configuration Parameters

### `num_stages` (default: 6)
Number of resolution stages to progress through

### `stage_duration` (default: 9.0)
Duration of each stage in seconds

### `use_fft_analysis` (default: False)
Enable FFT-based frequency analysis for resolution scheduling

### `images` (default: None)
List of training images for FFT analysis (required if `use_fft_analysis=True`)

### FFT-Specific Parameters (Auto-configured)
- `max_reso_scale`: Maximum downsampling factor (computed from images)
- `start_significance_factor`: Initial frequency threshold (default: 4)
- `reso_sample_num`: Number of resolution levels (default: 32)

## Reference Implementation

This implementation is based on the DashGaussian paper's training scheduler, adapted for DeadlineDinosaur's architecture with the following enhancements:

1. **Standalone FFT module**: Can be used independently
2. **Flexible downsampling**: Multiple methods with automatic fallback
3. **Backward compatibility**: Existing code continues to work
4. **Better defaults**: Improved interpolation methods (area vs bilinear)

## Future Enhancements

Potential improvements for future versions:

1. **Adaptive stage duration**: Adjust timing based on training loss
2. **Per-image resolution**: Different resolutions for different training images
3. **GPU Lanczos**: CUDA implementation for faster downsampling
4. **Resolution predictor**: ML model to predict optimal resolutions

## Troubleshooting

### FastLanczos Not Available
```
[ WARNING ] FastLanczos not available, falling back to torch interpolation
```
**Solution**: FastLanczos is optional. The fallback method still provides good quality.

### FFT Analysis Failed
```
[ WARNING ] FFT analysis requested but no images provided, falling back to time-based
```
**Solution**: Provide a list of training images when `use_fft_analysis=True`.

### Out of Memory
If FFT analysis causes OOM:
- Reduce `reso_sample_num` (default: 32)
- Process fewer images for analysis
- Use smaller image resolutions for FFT analysis

## Testing

Run the test suite to verify functionality:

```bash
python test_resolution_scheduler.py
```

Tests include:
- Basic time-based scheduler
- FFT-based scheduler with synthetic images
- High-quality downsampling (single and batch)
