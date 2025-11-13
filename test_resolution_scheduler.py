#!/usr/bin/env python3
"""
Test script for the enhanced ResolutionScheduler with FFT-based analysis.
"""
import torch
import numpy as np
import sys
import os

# Add the project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deadlinedino'))

from training.scheduling_utils import ResolutionScheduler

def test_basic_scheduler():
    """Test basic time-based scheduler without FFT analysis."""
    print("=" * 60)
    print("Test 1: Basic time-based scheduler")
    print("=" * 60)

    scheduler = ResolutionScheduler(num_stages=6, stage_duration=9.0)
    scheduler.start()

    # Test resolution scale
    scale = scheduler.get_resolution_scale()
    print(f"Initial resolution scale: {scale:.4f}")

    # Test downsampled shape
    height, width = scheduler.get_downsampled_shape(1080, 1920)
    print(f"Downsampled shape (1080x1920): {height}x{width}")

    # Test info dict
    info = scheduler.get_info_dict()
    print(f"Scheduler info: {info}")

    print("✓ Basic scheduler test passed\n")
    return True

def test_fft_scheduler():
    """Test FFT-based scheduler with synthetic images."""
    print("=" * 60)
    print("Test 2: FFT-based scheduler with synthetic images")
    print("=" * 60)

    # Create synthetic test images with different frequency content
    # Image 1: High frequency (checkerboard pattern)
    img1 = torch.zeros((3, 256, 256))
    for i in range(256):
        for j in range(256):
            if (i // 8 + j // 8) % 2 == 0:
                img1[:, i, j] = 1.0

    # Image 2: Low frequency (smooth gradient)
    img2 = torch.zeros((3, 256, 256))
    for i in range(256):
        for j in range(256):
            img2[:, i, j] = (i + j) / (256 * 2)

    # Image 3: Medium frequency
    img3 = torch.zeros((3, 256, 256))
    for i in range(256):
        for j in range(256):
            img3[:, i, j] = 0.5 + 0.5 * np.sin(i / 20.0) * np.sin(j / 20.0)

    images = [img1, img2, img3]

    try:
        scheduler = ResolutionScheduler(
            num_stages=6,
            stage_duration=9.0,
            use_fft_analysis=True,
            images=images
        )

        print(f"FFT scheduler initialized")
        print(f"Max resolution scale: {scheduler.max_reso_scale:.2f}")
        print(f"Number of resolution levels: {len(scheduler.reso_scales) if scheduler.reso_scales else 0}")

        # Start and test
        scheduler.start()
        scale = scheduler.get_resolution_scale()
        print(f"Initial resolution scale: {scale:.4f}")

        print("✓ FFT scheduler test passed\n")
        return True
    except Exception as e:
        print(f"✗ FFT scheduler test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_downsampling():
    """Test high-quality downsampling methods."""
    print("=" * 60)
    print("Test 3: High-quality downsampling")
    print("=" * 60)

    # Create a test image
    test_image = torch.rand((3, 512, 512))

    # Test single image downsampling
    try:
        downsampled = ResolutionScheduler.downsample_image_hq(
            test_image,
            target_height=256,
            target_width=256,
            use_lanczos=False  # Use torch fallback for testing
        )

        print(f"Input shape: {test_image.shape}")
        print(f"Downsampled shape: {downsampled.shape}")
        assert downsampled.shape == (3, 256, 256), "Downsampled shape incorrect"
        print("✓ Single image downsampling test passed")
    except Exception as e:
        print(f"✗ Single image downsampling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test batch downsampling
    try:
        batch_image = torch.rand((2, 3, 512, 512))
        downsampled_batch = ResolutionScheduler.downsample_image_hq(
            batch_image,
            target_height=256,
            target_width=256,
            use_lanczos=False
        )

        print(f"Batch input shape: {batch_image.shape}")
        print(f"Batch downsampled shape: {downsampled_batch.shape}")
        assert downsampled_batch.shape == (2, 3, 256, 256), "Batch downsampled shape incorrect"
        print("✓ Batch downsampling test passed")
    except Exception as e:
        print(f"✗ Batch downsampling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ResolutionScheduler Test Suite")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("Basic scheduler", test_basic_scheduler()))
    results.append(("FFT scheduler", test_fft_scheduler()))
    results.append(("Downsampling", test_downsampling()))

    # Print summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(passed for _, passed in results)
    print("=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
