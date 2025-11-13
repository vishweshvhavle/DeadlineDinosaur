#!/usr/bin/env python3
"""
Example usage of FFT-based ResolutionScheduler with training pipeline integration.

This demonstrates how to integrate the enhanced ResolutionScheduler into
your training loop with FFT analysis and high-quality downsampling.
"""

import torch
import numpy as np
import sys
import os
import time

# Add the project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deadlinedino'))

from training.scheduling_utils import ResolutionScheduler


def example_basic_usage():
    """
    Example 1: Basic time-based scheduler (existing behavior)
    """
    print("=" * 70)
    print("Example 1: Basic Time-Based Scheduler")
    print("=" * 70)

    # Create scheduler with default time-based behavior
    scheduler = ResolutionScheduler(
        num_stages=6,
        stage_duration=9.0
    )

    # Start the scheduler
    scheduler.start()

    # Simulate training loop
    for iteration in range(5):
        # Get current resolution scale
        scale = scheduler.get_resolution_scale()
        stage = scheduler.get_current_stage()

        # Get downsampled dimensions
        full_h, full_w = 1080, 1920
        ds_h, ds_w = scheduler.get_downsampled_shape(full_h, full_w)

        print(f"Iteration {iteration}: Stage {stage}/6, Scale {scale:.3f}, "
              f"Resolution {ds_h}x{ds_w}")

        time.sleep(1)  # Simulate training time

    print()


def example_fft_based_scheduler():
    """
    Example 2: FFT-based scheduler with frequency analysis
    """
    print("=" * 70)
    print("Example 2: FFT-Based Scheduler with Frequency Analysis")
    print("=" * 70)

    # Create synthetic training images with different frequency content
    print("Creating synthetic training images...")

    # Image 1: High frequency (checkerboard pattern)
    img1 = torch.zeros((3, 512, 512), device='cuda')
    for i in range(512):
        for j in range(512):
            if (i // 16 + j // 16) % 2 == 0:
                img1[:, i, j] = 1.0

    # Image 2: Medium frequency (sine waves)
    img2 = torch.zeros((3, 512, 512), device='cuda')
    x = torch.arange(512, device='cuda').float()
    y = torch.arange(512, device='cuda').float()
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    img2 = 0.5 + 0.5 * torch.sin(xx / 30.0) * torch.sin(yy / 30.0)
    img2 = img2.unsqueeze(0).repeat(3, 1, 1)

    # Image 3: Low frequency (smooth gradient)
    img3 = torch.zeros((3, 512, 512), device='cuda')
    for i in range(512):
        for j in range(512):
            img3[:, i, j] = (i + j) / (512 * 2)

    images = [img1, img2, img3]
    print(f"Created {len(images)} synthetic images")

    # Create FFT-based scheduler
    print("\nInitializing FFT-based scheduler...")
    scheduler = ResolutionScheduler(
        num_stages=6,
        stage_duration=9.0,
        use_fft_analysis=True,
        images=images
    )

    # Print schedule information
    if scheduler.reso_scales:
        print(f"\nResolution Schedule:")
        print(f"  Total levels: {len(scheduler.reso_scales)}")
        print(f"  Initial scale: 1/{scheduler.reso_scales[0]:.2f}")
        print(f"  Final scale: 1/{scheduler.reso_scales[-1]:.2f}")
        print(f"\nFirst 10 resolution levels:")
        for i in range(min(10, len(scheduler.reso_scales))):
            if i < len(scheduler.reso_level_begin):
                print(f"    Level {i}: 1/{scheduler.reso_scales[i]:.2f} "
                      f"at {scheduler.reso_level_begin[i]}s")

    # Start scheduler and simulate training
    print("\nSimulating training with FFT-based scheduling:")
    scheduler.start()

    for iteration in range(5):
        scale = scheduler.get_resolution_scale()
        ds_h, ds_w = scheduler.get_downsampled_shape(1080, 1920)

        print(f"Iteration {iteration}: Scale {scale:.4f}, "
              f"Resolution {ds_h}x{ds_w}")

        time.sleep(1)

    print()


def example_high_quality_downsampling():
    """
    Example 3: High-quality image downsampling
    """
    print("=" * 70)
    print("Example 3: High-Quality Image Downsampling")
    print("=" * 70)

    # Create test image
    print("Creating test image (1080x1920)...")
    test_image = torch.rand((3, 1080, 1920), device='cuda')

    # Downsample using high-quality method
    print("Downsampling to 540x960...")

    # Try with Lanczos first (will fallback if not available)
    downsampled = ResolutionScheduler.downsample_image_hq(
        test_image,
        target_height=540,
        target_width=960,
        use_lanczos=True
    )

    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {downsampled.shape}")
    print(f"Memory usage: {test_image.element_size() * test_image.nelement() / 1024 / 1024:.2f} MB -> "
          f"{downsampled.element_size() * downsampled.nelement() / 1024 / 1024:.2f} MB")

    # Batch downsampling
    print("\nBatch downsampling (4 images)...")
    batch_images = torch.rand((4, 3, 1080, 1920), device='cuda')

    batch_downsampled = ResolutionScheduler.downsample_image_hq(
        batch_images,
        target_height=540,
        target_width=960,
        use_lanczos=False  # Use torch fallback for batch
    )

    print(f"Batch input shape: {batch_images.shape}")
    print(f"Batch output shape: {batch_downsampled.shape}")

    print()


def example_training_integration():
    """
    Example 4: Integration with training loop
    """
    print("=" * 70)
    print("Example 4: Training Loop Integration")
    print("=" * 70)

    # Setup scheduler
    scheduler = ResolutionScheduler(num_stages=6, stage_duration=9.0)
    scheduler.start()

    # Simulate training loop
    print("Simulating training loop with progressive resolution:")

    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}")

        # Get current resolution
        scale = scheduler.get_resolution_scale()
        info = scheduler.get_info_dict()

        # Original camera intrinsics (example)
        full_height, full_width = 1080, 1920
        focal_x, focal_y = 1000.0, 1000.0

        # Get downsampled resolution
        ds_height, ds_width = scheduler.get_downsampled_shape(
            full_height, full_width
        )

        # Adjust projection matrix for center crop rendering
        proj_matrix = np.array([
            [focal_x, 0, 0, 0],
            [0, focal_y, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ], dtype=np.float32).T

        adjusted_proj = scheduler.get_downsampled_proj_matrix(
            proj_matrix, full_height, full_width
        )

        print(f"  Resolution: {ds_height}x{ds_width} (scale: {scale:.3f})")
        print(f"  Original focal: ({focal_x:.1f}, {focal_y:.1f})")
        print(f"  Adjusted focal: ({adjusted_proj[0, 0]:.1f}, {adjusted_proj[1, 1]:.1f})")
        print(f"  Elapsed time: {info['elapsed_time']:.2f}s")

        # Simulate rendering and training
        # rendered_image = render(..., proj_matrix=adjusted_proj,
        #                         output_size=(ds_height, ds_width))

        # Downsample ground truth to match
        # gt_image = torch.rand((1, 3, full_height, full_width), device='cuda')
        # gt_downsampled = scheduler.downsample_image_hq(
        #     gt_image, ds_height, ds_width, use_lanczos=True
        # )

        # Compute loss
        # loss = compute_loss(rendered_image, gt_downsampled)

        time.sleep(1)

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Enhanced ResolutionScheduler Examples")
    print("FFT-Based Analysis & High-Quality Downsampling")
    print("=" * 70 + "\n")

    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("WARNING: CUDA not available, using CPU")
        print()

        # Run examples
        example_basic_usage()

        if torch.cuda.is_available():
            example_fft_based_scheduler()
            example_high_quality_downsampling()
        else:
            print("Skipping GPU examples (CUDA not available)\n")

        example_training_integration()

        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
