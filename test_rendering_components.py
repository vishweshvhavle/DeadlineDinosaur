"""
Test script to verify each component of the Gaussian Splatting pipeline
Run this to identify which component is causing black images
"""
import torch
import numpy as np
import sys
import os
from PIL import Image

import deadlinedino
import deadlinedino.io_manager as io_manager
import deadlinedino.scene as scene
from deadlinedino.utils import rgb_to_sh0

def test_colmap_loading(source_path):
    """Test if COLMAP data is loading correctly"""
    print("\n" + "="*80)
    print("TEST 1: COLMAP Data Loading")
    print("="*80)

    try:
        cameras_info, camera_frames, init_xyz, init_color = io_manager.load_colmap_result(source_path, "images")

        print(f"✓ Successfully loaded COLMAP data")
        print(f"  - Number of points: {init_xyz.shape[0]}")
        print(f"  - Number of cameras: {len(cameras_info)}")
        print(f"  - Number of frames: {len(camera_frames)}")

        print(f"\n  XYZ range:")
        print(f"    Min: {init_xyz.min(axis=0)}")
        print(f"    Max: {init_xyz.max(axis=0)}")
        print(f"    Mean: {init_xyz.mean(axis=0)}")

        print(f"\n  Color range (should be [0,1]):")
        print(f"    Min: {init_color.min(axis=0)}")
        print(f"    Max: {init_color.max(axis=0)}")
        print(f"    Mean: {init_color.mean(axis=0)}")

        if init_xyz.shape[0] == 0:
            print("  ✗ ERROR: No points loaded!")
            return False

        if init_color.min() < 0 or init_color.max() > 1:
            print("  ⚠ WARNING: Colors outside [0,1] range")

        if init_color.mean() < 0.01:
            print("  ✗ ERROR: Point cloud colors are nearly black!")
            return False

        print("  ✓ COLMAP data looks good")
        return True

    except Exception as e:
        print(f"  ✗ ERROR loading COLMAP data: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gaussian_initialization(init_xyz, init_color, sh_degree=3):
    """Test if Gaussian parameters are initialized correctly"""
    print("\n" + "="*80)
    print("TEST 2: Gaussian Initialization")
    print("="*80)

    try:
        init_xyz_t = torch.tensor(init_xyz, dtype=torch.float32, device='cuda')
        init_color_t = torch.tensor(init_color, dtype=torch.float32, device='cuda')

        xyz, scale, rot, sh_0, sh_rest, opacity = scene.create_gaussians(init_xyz_t, init_color_t, sh_degree)

        print(f"✓ Gaussian parameters initialized")
        print(f"  - Number of Gaussians: {xyz.shape[-1]}")

        def print_stats(name, tensor):
            t = tensor.detach().cpu()
            print(f"\n  {name}:")
            print(f"    Shape: {tensor.shape}")
            print(f"    Min: {t.min().item():.6f}, Max: {t.max().item():.6f}")
            print(f"    Mean: {t.mean().item():.6f}, Std: {t.std().item():.6f}")

            if torch.isnan(t).any():
                print(f"    ✗ ERROR: Contains NaN!")
                return False
            if torch.isinf(t).any():
                print(f"    ✗ ERROR: Contains Inf!")
                return False
            return True

        all_ok = True
        all_ok &= print_stats("XYZ", xyz)
        all_ok &= print_stats("Scale (log)", scale)
        all_ok &= print_stats("Rotation", rot)
        all_ok &= print_stats("SH_0 (base color)", sh_0)
        all_ok &= print_stats("Opacity (logit)", opacity)

        # Check activated values
        print(f"\n  Activated values:")
        print(f"    Opacity (sigmoid): min={opacity.sigmoid().min().item():.4f}, max={opacity.sigmoid().max().item():.4f}, mean={opacity.sigmoid().mean().item():.4f}")
        print(f"    Scale (exp): min={scale.exp().min().item():.4f}, max={scale.exp().max().item():.4f}, mean={scale.exp().mean().item():.4f}")

        if opacity.sigmoid().mean().item() < 0.01:
            print(f"  ✗ ERROR: Average opacity too low!")
            all_ok = False

        # Test SH to RGB conversion
        print(f"\n  Testing SH to RGB conversion...")
        test_color = init_color_t[:100].transpose(0, 1)
        test_sh0 = rgb_to_sh0(test_color)
        print(f"    Input RGB range: [{test_color.min().item():.3f}, {test_color.max().item():.3f}]")
        print(f"    SH_0 range: [{test_sh0.min().item():.3f}, {test_sh0.max().item():.3f}]")
        print(f"    SH_0 mean: {test_sh0.mean().item():.3f}")

        if all_ok:
            print("\n  ✓ Gaussian initialization looks good")
        else:
            print("\n  ✗ Gaussian initialization has issues")

        return all_ok, (xyz, scale, rot, sh_0, sh_rest, opacity)

    except Exception as e:
        print(f"  ✗ ERROR in Gaussian initialization: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_sh_to_rgb_conversion(sh_0, sh_rest, sh_degree=3):
    """Test spherical harmonics to RGB conversion"""
    print("\n" + "="*80)
    print("TEST 3: SH to RGB Conversion")
    print("="*80)

    try:
        from deadlinedino.utils.wrapper import SphericalHarmonicToRGB

        # Create test directions (random unit vectors)
        n_test = min(1000, sh_0.shape[-1])
        dirs = torch.randn((1, 3, n_test), device='cuda')
        dirs = torch.nn.functional.normalize(dirs, dim=1)

        # Convert SH to RGB
        test_sh0 = sh_0[:, :, :n_test]
        test_sh_rest = sh_rest[:, :, :n_test]

        rgb = SphericalHarmonicToRGB.call_fused(sh_degree, test_sh0, test_sh_rest, dirs)

        print(f"✓ SH to RGB conversion executed")
        print(f"  Input SH_0 range: [{test_sh0.min().item():.3f}, {test_sh0.max().item():.3f}]")
        print(f"  Output RGB range: [{rgb.min().item():.3f}, {rgb.max().item():.3f}]")
        print(f"  Output RGB mean: {rgb.mean().item():.3f}")

        if rgb.mean().item() < 0.01:
            print(f"  ✗ ERROR: RGB output is nearly black!")
            return False

        if rgb.min().item() < -0.1 or rgb.max().item() > 1.1:
            print(f"  ⚠ WARNING: RGB values outside expected [0,1] range")

        print("  ✓ SH to RGB conversion looks good")
        return True

    except Exception as e:
        print(f"  ✗ ERROR in SH to RGB conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_rendering_components.py <scene_path>")
        print("Example: python test_rendering_components.py data/test/1748243104741")
        sys.exit(1)

    source_path = sys.argv[1]

    print("\n" + "="*80)
    print("GAUSSIAN SPLATTING COMPONENT TEST")
    print("="*80)
    print(f"Scene: {source_path}")

    # Test 1: COLMAP loading
    if not test_colmap_loading(source_path):
        print("\n" + "="*80)
        print("FAILED: COLMAP loading test")
        print("="*80)
        return

    # Reload for next tests
    cameras_info, camera_frames, init_xyz, init_color = io_manager.load_colmap_result(source_path, "images")

    # Test 2: Gaussian initialization
    success, gaussian_params = test_gaussian_initialization(init_xyz, init_color)
    if not success:
        print("\n" + "="*80)
        print("FAILED: Gaussian initialization test")
        print("="*80)
        return

    xyz, scale, rot, sh_0, sh_rest, opacity = gaussian_params

    # Test 3: SH to RGB conversion
    if not test_sh_to_rgb_conversion(sh_0, sh_rest):
        print("\n" + "="*80)
        print("FAILED: SH to RGB conversion test")
        print("="*80)
        return

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\nThe rendering pipeline components appear to be working correctly.")
    print("If you're still seeing black images, the issue may be in:")
    print("  - Training dynamics (learning rates, loss computation)")
    print("  - Resolution scaling (DashGaussian progressive training)")
    print("  - View/projection matrix computation")
    print("\nRun the main training with debug mode enabled to see more details:")
    print("  python example_train.py -s <scene> -m <output> --iterations 100")


if __name__ == "__main__":
    main()
