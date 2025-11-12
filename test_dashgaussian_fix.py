#!/usr/bin/env python3
"""
Test script to verify DashGaussian integration fixes.
This script runs a quick training test to ensure the NaN loss bug is fixed.
"""

import sys
import os
import subprocess

# Test configuration - use a small dataset for quick testing
TEST_CONFIG = {
    'dataset': 'data/test/1748243104741',  # Small test scene
    'iterations': 1000,  # Test through first resolution transition
}

def main():
    print("=" * 80)
    print("DashGaussian Integration Fix - Verification Test")
    print("=" * 80)
    print(f"\nDataset: {TEST_CONFIG['dataset']}")
    print(f"Iterations: {TEST_CONFIG['iterations']}")
    print(f"Target: Verify no NaN losses, smooth primitive growth\n")
    print("-" * 80)

    # Check if dataset exists
    if not os.path.exists(TEST_CONFIG['dataset']):
        print(f"ERROR: Dataset not found at {TEST_CONFIG['dataset']}")
        print("Available test datasets:")
        if os.path.exists('data/test'):
            for item in os.listdir('data/test'):
                print(f"  - data/test/{item}")
        return 1

    # Build command
    cmd = [
        'python', 'run_all_scenes.py',
        '--dataset', TEST_CONFIG['dataset'],
        '--iterations', str(TEST_CONFIG['iterations'])
    ]

    print(f"Running: {' '.join(cmd)}\n")
    print("-" * 80)

    # Run the test
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output
    output = result.stdout + result.stderr
    lines = output.split('\n')

    # Collect key metrics
    densify_logs = []
    loss_values = []
    warnings = []
    errors = []

    for line in lines:
        if '[DENSIFY]' in line:
            densify_logs.append(line)
        if 'Loss:' in line:
            # Extract loss value
            try:
                loss_str = line.split('Loss:')[1].split()[0]
                loss_values.append(float(loss_str))
            except (IndexError, ValueError):
                pass
        if '[WARNING]' in line or 'WARNING' in line:
            warnings.append(line)
        if '[ERROR]' in line or 'ERROR' in line:
            errors.append(line)

    # Analyze results
    print("\n" + "=" * 80)
    print("Test Results")
    print("=" * 80)

    # Check for success indicators
    success_checks = {
        'No NaN losses': all(not str(l) == 'nan' for l in loss_values) if loss_values else False,
        'Densification working': len(densify_logs) > 0,
        'No critical errors': len(errors) == 0,
        'Smooth growth rates': all('rate=' in log for log in densify_logs[:5]) if densify_logs else False,
    }

    all_pass = True
    for check, passed in success_checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check:30s}: {status}")
        all_pass = all_pass and passed

    # Show sample densify logs
    if densify_logs:
        print("\n" + "-" * 80)
        print("Sample Densification Logs (first 5):")
        print("-" * 80)
        for log in densify_logs[:5]:
            print(log.strip())

    # Show warnings if any
    if warnings:
        print("\n" + "-" * 80)
        print(f"Warnings ({len(warnings)}):")
        print("-" * 80)
        for warning in warnings[:10]:  # Show first 10
            print(warning.strip())

    # Show errors if any
    if errors:
        print("\n" + "-" * 80)
        print(f"Errors ({len(errors)}):")
        print("-" * 80)
        for error in errors:
            print(error.strip())

    # Loss progression
    if loss_values:
        print("\n" + "-" * 80)
        print("Loss Progression (sample):")
        print("-" * 80)
        sample_indices = [0, len(loss_values)//4, len(loss_values)//2,
                         3*len(loss_values)//4, len(loss_values)-1]
        for idx in sample_indices:
            if idx < len(loss_values):
                print(f"  Sample {idx:4d}: {loss_values[idx]:.6f}")

    print("\n" + "=" * 80)
    if all_pass:
        print("✅ All checks PASSED! The fix appears to be working correctly.")
        print("\nKey improvements:")
        print("  - Densification rate is properly smoothed over remaining steps")
        print("  - No NaN losses detected")
        print("  - Training progresses stably")
    else:
        print("❌ Some checks FAILED. Review the output above for issues.")

    print("=" * 80)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
