"""
Quick script to inspect COLMAP point cloud and verify it has colors
"""
import sys
import numpy as np
import os
from plyfile import PlyData

def inspect_pointcloud(scene_path):
    ply_path = os.path.join(scene_path, "sparse/0/points3D.ply")

    if not os.path.exists(ply_path):
        print(f"Error: {ply_path} does not exist")
        return

    print(f"Reading: {ply_path}")
    ply_data = PlyData.read(ply_path)
    vertices = ply_data['vertex']

    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T

    print(f"\nPoint Cloud Statistics:")
    print(f"  Number of points: {positions.shape[0]}")

    print(f"\n  Position (XYZ) range:")
    print(f"    Min:  {positions.min(axis=0)}")
    print(f"    Max:  {positions.max(axis=0)}")
    print(f"    Mean: {positions.mean(axis=0)}")

    print(f"\n  Color (RGB) range [should be 0-255]:")
    print(f"    Min:  {colors.min(axis=0)}")
    print(f"    Max:  {colors.max(axis=0)}")
    print(f"    Mean: {colors.mean(axis=0)}")

    # Normalized colors
    colors_norm = colors / 255.0
    print(f"\n  Normalized color (RGB) range [should be 0-1]:")
    print(f"    Min:  {colors_norm.min(axis=0)}")
    print(f"    Max:  {colors_norm.max(axis=0)}")
    print(f"    Mean: {colors_norm.mean(axis=0)}")

    # Check for issues
    print(f"\n  Analysis:")
    if positions.shape[0] == 0:
        print("    ✗ ERROR: No points in point cloud!")
    elif positions.shape[0] < 100:
        print(f"    ⚠ WARNING: Very few points ({positions.shape[0]})")
    else:
        print(f"    ✓ Point count looks good ({positions.shape[0]} points)")

    if colors.mean() < 1.0:
        print("    ✗ ERROR: Colors are nearly black! Check COLMAP reconstruction")
    elif colors_norm.mean() < 0.1:
        print("    ⚠ WARNING: Colors are very dark (mean < 0.1)")
    else:
        print(f"    ✓ Colors look reasonable (mean = {colors_norm.mean():.3f})")

    # Sample some points
    print(f"\n  Sample points (first 5):")
    for i in range(min(5, positions.shape[0])):
        pos = positions[i]
        col = colors[i]
        print(f"    Point {i}: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
              f"color=[{col[0]:.0f}, {col[1]:.0f}, {col[2]:.0f}]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_pointcloud.py <scene_path>")
        print("Example: python inspect_pointcloud.py data/test/1748243104741")
        sys.exit(1)

    inspect_pointcloud(sys.argv[1])
