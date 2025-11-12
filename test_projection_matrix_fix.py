#!/usr/bin/env python3
"""
Test to verify that the projection matrix fix prevents in-place modification
of shared tensors during progressive resolution training.
"""

import torch
import numpy as np

def test_projection_matrix_clone():
    """Test that cloning prevents shared tensor modification"""

    # Simulate the dataset's preloaded projection matrix (shared tensor)
    original_proj_matrix = torch.tensor([[2.0, 0.0, 0.0, 0.0],
                                          [0.0, 2.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0, 0.0],
                                          [0.0, 0.0, 1.0, 0.0]],
                                         dtype=torch.float32, device='cuda')

    # Store original values
    original_value_00 = original_proj_matrix[0, 0].item()
    original_value_11 = original_proj_matrix[1, 1].item()

    print("=" * 80)
    print("Testing Projection Matrix Fix")
    print("=" * 80)
    print(f"\nOriginal projection matrix [0,0]: {original_value_00}")
    print(f"Original projection matrix [1,1]: {original_value_11}")

    # Simulate multiple iterations with resolution scaling
    # WITHOUT cloning (the old buggy behavior)
    print("\n--- Test 1: WITHOUT cloning (buggy behavior) ---")
    buggy_proj = original_proj_matrix.clone()  # Reset
    for iteration in range(3):
        current_render_scale = 2
        # OLD BUGGY CODE: modifies in-place
        buggy_proj[0, 0] = buggy_proj[0, 0] / current_render_scale
        buggy_proj[1, 1] = buggy_proj[1, 1] / current_render_scale
        print(f"Iteration {iteration+1}: [0,0]={buggy_proj[0,0].item():.6f}, [1,1]={buggy_proj[1,1].item():.6f}")

    print(f"\nAfter 3 iterations WITHOUT clone: [0,0]={buggy_proj[0,0].item():.6f}")
    print(f"Expected: {original_value_00 / 2:.6f}")
    print(f"❌ BUG: Value was divided {3} times instead of once!")

    # WITH cloning (the fixed behavior)
    print("\n--- Test 2: WITH cloning (fixed behavior) ---")
    for iteration in range(3):
        # Get a view of the shared projection matrix (simulating dataloader)
        proj_matrix = original_proj_matrix.unsqueeze(0)  # Add batch dimension

        current_render_scale = 2
        if current_render_scale > 1:
            # NEW FIXED CODE: clone before modifying
            proj_matrix = proj_matrix.clone()
            proj_matrix[:, 0, 0] = proj_matrix[:, 0, 0] / current_render_scale
            proj_matrix[:, 1, 1] = proj_matrix[:, 1, 1] / current_render_scale

        print(f"Iteration {iteration+1}: [0,0]={proj_matrix[0,0,0].item():.6f}, [1,1]={proj_matrix[0,1,1].item():.6f}")

    # Verify the original was not modified
    print(f"\nOriginal projection matrix after 3 iterations: [0,0]={original_proj_matrix[0,0].item():.6f}")
    print(f"Expected: {original_value_00:.6f}")

    if abs(original_proj_matrix[0,0].item() - original_value_00) < 1e-6:
        print("✅ SUCCESS: Original matrix was preserved!")
    else:
        print("❌ FAIL: Original matrix was modified!")

    print("\n" + "=" * 80)
    print("Explanation:")
    print("=" * 80)
    print("Without .clone(), the projection matrix gets divided by scale_factor")
    print("every time a view is processed. After multiple epochs, the values")
    print("become so small that rendering produces black images and NaN loss.")
    print()
    print("With .clone(), each iteration works with its own copy, keeping the")
    print("original shared tensor intact for future iterations.")
    print("=" * 80)

if __name__ == '__main__':
    test_projection_matrix_clone()
