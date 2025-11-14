#!/usr/bin/env python3
"""
Test script for the new TrainingScheduler with momentum-based budgeting
and iteration-based frequency resolution scheduling.
"""

import torch
import numpy as np
import sys
import os

# Add deadlinedino to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deadlinedino'))

from training.scheduling_utils import TrainingScheduler
from arguments import OptimizationParams, PipelineParams


class MockGaussianModel:
    """Mock GaussianModel for testing."""
    def __init__(self, n_points=10000):
        self.get_xyz = torch.randn(n_points, 3).cuda()


def create_test_images(num_images=5, height=256, width=256):
    """Create synthetic test images with varying frequency content."""
    images = []
    for i in range(num_images):
        # Create images with different frequency characteristics
        img = torch.randn(3, height, width).cuda()
        # Add some structure
        x = torch.linspace(-1, 1, width).cuda()
        y = torch.linspace(-1, 1, height).cuda()
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        # Add some sinusoidal patterns with different frequencies
        freq = (i + 1) * 2
        pattern = torch.sin(freq * xx) * torch.cos(freq * yy)
        img = img * 0.3 + pattern.unsqueeze(0) * 0.7

        images.append(img)

    return images


def test_resolution_scheduler():
    """Test the resolution scheduler functionality."""
    print("\n" + "="*80)
    print("Testing TrainingScheduler - Resolution Scheduling")
    print("="*80)

    # Create mock parameters
    opt = OptimizationParams.get_class_default_obj()
    opt.iterations = 30000
    opt.densify_until_iter = 15000
    opt.densification_interval = 100

    pipe = PipelineParams.get_class_default_obj()
    pipe.resolution_mode = 'freq'
    pipe.densify_mode = 'freq'
    pipe.max_n_gaussian = -1  # Auto mode

    # Create mock Gaussian model
    gaussians = MockGaussianModel(n_points=10000)

    # Create test images
    print("\n[ INFO ] Creating test images...")
    images = create_test_images(num_images=5, height=256, width=256)

    # Initialize scheduler
    print("\n[ INFO ] Initializing TrainingScheduler...")
    scheduler = TrainingScheduler(opt, pipe, gaussians, images)

    # Test resolution scale at different iterations
    print("\n[ INFO ] Resolution scale progression:")
    print("-" * 80)
    print(f"{'Iteration':<12} {'Scale':<10} {'Inverse (1/scale)':<20} {'Resolution %':<15}")
    print("-" * 80)

    test_iterations = [0, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 30000]
    for iteration in test_iterations:
        scale = scheduler.get_res_scale(iteration)
        inverse_scale = scheduler.get_resolution_scale_inverse(iteration)
        resolution_pct = inverse_scale * 100
        print(f"{iteration:<12} {scale:<10} {inverse_scale:<20.4f} {resolution_pct:<15.2f}%")

    print("\n" + "="*80)


def test_momentum_budgeting():
    """Test momentum-based primitive budgeting."""
    print("\n" + "="*80)
    print("Testing TrainingScheduler - Momentum-based Primitive Budgeting")
    print("="*80)

    # Create mock parameters
    opt = OptimizationParams.get_class_default_obj()
    opt.iterations = 30000
    opt.densify_until_iter = 15000
    opt.densification_interval = 100

    pipe = PipelineParams.get_class_default_obj()
    pipe.resolution_mode = 'freq'
    pipe.densify_mode = 'freq'
    pipe.max_n_gaussian = -1  # Auto mode

    # Create mock Gaussian model
    gaussians = MockGaussianModel(n_points=10000)

    # Create test images
    images = create_test_images(num_images=3, height=128, width=128)

    # Initialize scheduler
    scheduler = TrainingScheduler(opt, pipe, gaussians, images)

    print(f"\n[ INFO ] Initial Gaussian count: {scheduler.init_n_gaussian}")
    print(f"[ INFO ] Initial momentum: {scheduler.momentum}")
    print(f"[ INFO ] Initial max Gaussians: {scheduler.max_n_gaussian}")

    # Simulate densification steps
    print("\n[ INFO ] Simulating densification with momentum updates:")
    print("-" * 80)
    print(f"{'Iteration':<12} {'Cur Gaussians':<15} {'Scale':<8} {'Densify Rate':<15} {'Momentum':<12} {'Max Gaussians':<15}")
    print("-" * 80)

    current_gaussians = scheduler.init_n_gaussian
    test_iterations = [100, 500, 1000, 2000, 5000, 7500, 10000, 12500, 15000]

    for iteration in test_iterations:
        scale = scheduler.get_res_scale(iteration)
        densify_rate = scheduler.get_densify_rate(iteration, current_gaussians, scale)

        # Simulate adding Gaussians based on densify rate
        added_gaussians = int(current_gaussians * densify_rate)

        # Update momentum
        scheduler.update_momentum(added_gaussians)

        # Update current Gaussian count
        current_gaussians += added_gaussians

        print(f"{iteration:<12} {current_gaussians:<15} {scale:<8} {densify_rate:<15.4f} {scheduler.momentum:<12} {scheduler.max_n_gaussian:<15}")

    print("\n" + "="*80)


def test_fixed_budget_mode():
    """Test with fixed maximum Gaussian budget."""
    print("\n" + "="*80)
    print("Testing TrainingScheduler - Fixed Budget Mode")
    print("="*80)

    # Create mock parameters
    opt = OptimizationParams.get_class_default_obj()
    opt.iterations = 30000
    opt.densify_until_iter = 15000

    pipe = PipelineParams.get_class_default_obj()
    pipe.resolution_mode = 'freq'
    pipe.densify_mode = 'freq'
    pipe.max_n_gaussian = 100000  # Fixed budget

    # Create mock Gaussian model
    gaussians = MockGaussianModel(n_points=10000)

    # Create test images
    images = create_test_images(num_images=3, height=128, width=128)

    # Initialize scheduler
    scheduler = TrainingScheduler(opt, pipe, gaussians, images)

    print(f"\n[ INFO ] Initial Gaussian count: {scheduler.init_n_gaussian}")
    print(f"[ INFO ] Max Gaussians (fixed): {scheduler.max_n_gaussian}")
    print(f"[ INFO ] Momentum mode: {'Auto' if scheduler.momentum == -1 else 'Active'}")

    print("\n" + "="*80)


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " " * 20 + "TrainingScheduler Test Suite" + " " * 30 + "║")
    print("╚" + "="*78 + "╝")

    # Run tests
    test_resolution_scheduler()
    test_momentum_budgeting()
    test_fixed_budget_mode()

    print("\n" + "="*80)
    print("All tests completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
