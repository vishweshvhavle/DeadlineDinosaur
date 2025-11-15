#!/usr/bin/env python3
"""
Convert SLAM dataset to 3D Gaussian Splatting compatible format
Supports batch processing of multiple datasets with automatic validation
"""


import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def load_video_info(video_info_path):
    """Load frame ID to timestamp mapping"""
    frame_to_timestamp = {}
    with open(video_info_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                frame_id = int(parts[0])
                timestamp = parts[1]
                frame_to_timestamp[frame_id] = timestamp
    return frame_to_timestamp


def extract_video_frames(video_path, output_dir, frame_to_timestamp):
    """Extract frames from video with proper naming"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return False, set()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_id = 1  # Start from 1 to match videoInfo.txt
    extracted = 0
    extracted_filenames = set()
    
    print(f"  Extracting {total_frames} frames...")
    with tqdm(total=total_frames, leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get timestamp for this frame
            if frame_id in frame_to_timestamp:
                timestamp = frame_to_timestamp[frame_id]
                frame_name = f"{timestamp}.jpg"
                output_path = os.path.join(output_dir, frame_name)
                
                # Try to write the frame
                success = cv2.imwrite(output_path, frame)
                if success:
                    extracted += 1
                    extracted_filenames.add(frame_name)
                else:
                    print(f"  ⚠ Warning: Failed to write frame {frame_name}")
            
            frame_id += 1
            pbar.update(1)
    
    cap.release()
    print(f"  ✓ Extracted {extracted} frames")
    return True, extracted_filenames


def verify_colmap_format(slam_dir):
    """Verify COLMAP files exist and check basic format"""
    required_files = ['cameras.txt', 'images.txt', 'points3D.txt']
    
    for filename in required_files:
        filepath = os.path.join(slam_dir, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: Missing {filename}")
            return False
    
    # Check images.txt format
    images_file = os.path.join(slam_dir, 'images.txt')
    with open(images_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                print(f"  WARNING: images.txt may have incorrect format")
                print(f"  Expected: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME")
                return False
            break
    
    return True


def process_images_txt(input_path, output_path, extracted_filenames):
    """
    Process images.txt to ensure correct format and filter out missing images
    Only include entries for images that were actually extracted
    
    Args:
        input_path: Path to original images.txt
        output_path: Path to output images.txt
        extracted_filenames: Set of filenames that were successfully extracted
    """
    total_entries = 0
    kept_entries = 0
    removed_entries = []
    
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        # Write COLMAP header
        f_out.write("# Image list with two lines of data per image:\n")
        f_out.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f_out.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        skip_next_line = False
        
        for line in f_in:
            # Skip comments but not empty lines (they're part of COLMAP format)
            if line.startswith('#'):
                continue
            
            # Handle empty lines (second line of each entry)
            if not line.strip():
                if not skip_next_line:
                    f_out.write("\n")
                skip_next_line = False
                continue
            
            # If we're skipping due to missing image, skip this line too
            if skip_next_line:
                skip_next_line = False
                continue
            
            parts = line.strip().split()
            
            # Process image entry line
            if len(parts) >= 10:
                total_entries += 1
                image_id = parts[0]
                qw, qx, qy, qz = parts[1:5]
                tx, ty, tz = parts[5:8]
                camera_id = parts[8]
                image_name = parts[9]  # Should be timestamp.jpg
                
                # Check if this image was actually extracted
                if image_name in extracted_filenames:
                    # Write in standard COLMAP format
                    f_out.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n")
                    kept_entries += 1
                else:
                    # Skip this entry and its feature line
                    removed_entries.append((image_id, image_name))
                    skip_next_line = True
    
    print(f"  ✓ Processed images.txt:")
    print(f"    - Total entries in original: {total_entries}")
    print(f"    - Kept entries: {kept_entries}")
    print(f"    - Removed entries: {len(removed_entries)}")
    
    if removed_entries:
        print(f"    ⚠ Removed {len(removed_entries)} entries for missing images:")
        for img_id, img_name in removed_entries[:5]:  # Show first 5
            print(f"      - ID {img_id}: {img_name}")
        if len(removed_entries) > 5:
            print(f"      ... and {len(removed_entries) - 5} more")
    
    return kept_entries > 0


def validate_output(images_dir, sparse_dir):
    """
    Validate that all images referenced in images.txt exist
    Returns True if validation passes, False otherwise
    """
    print("\n[Validation] Checking output consistency...")
    
    # Get list of actual image files
    images_path = Path(images_dir)
    actual_images = set([f.name for f in images_path.glob("*.jpg")])
    
    # Parse images.txt to get referenced images
    images_txt_path = sparse_dir / "images.txt"
    referenced_images = set()
    
    with open(images_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 10:
                image_name = parts[9]
                referenced_images.add(image_name)
    
    # Find mismatches
    missing_images = referenced_images - actual_images
    unreferenced_images = actual_images - referenced_images
    
    print(f"  - Images on disk: {len(actual_images)}")
    print(f"  - Images in images.txt: {len(referenced_images)}")
    
    if missing_images:
        print(f"  ✗ ERROR: {len(missing_images)} images referenced but not found on disk:")
        for img in list(missing_images)[:5]:
            print(f"      - {img}")
        return False
    
    if unreferenced_images:
        print(f"  ⚠ Warning: {len(unreferenced_images)} images on disk but not in images.txt")
        print(f"    (This is OK - these images won't be used in training)")
    
    print(f"  ✓ Validation passed: All referenced images exist")
    return True


def convert_dataset(dataset_root, output_root):
    """
    Main conversion function with validation
    
    Args:
        dataset_root: Path to input dataset (e.g., /path/to/1748153841908)
        output_root: Path to output directory
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    
    dataset_name = dataset_root.name
    print(f"\n{'='*60}")
    print(f"Converting: {dataset_name}")
    print(f"{'='*60}")
    
    # Create output structure
    images_dir = output_root / "images"
    sparse_dir = output_root / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths
    video_path = dataset_root / f"{dataset_name}_flip.mp4"
    inputs_dir = dataset_root / "inputs"
    slam_dir = inputs_dir / "slam"
    video_info_path = inputs_dir / "videoInfo.txt"
    
    # Check if all required files exist
    if not video_path.exists():
        print(f"  ✗ ERROR: Video not found at {video_path}")
        return False
    
    if not inputs_dir.exists() or not slam_dir.exists():
        print(f"  ✗ ERROR: Required directories not found")
        return False
    
    if not video_info_path.exists():
        print(f"  ✗ ERROR: videoInfo.txt not found")
        return False
    
    if not verify_colmap_format(slam_dir):
        print("  ✗ ERROR: COLMAP files are missing or malformed")
        return False
    
    # Step 1: Load video info
    print("\n[1/5] Loading video information...")
    try:
        frame_to_timestamp = load_video_info(video_info_path)
        print(f"  ✓ Found {len(frame_to_timestamp)} frames in videoInfo.txt")
    except Exception as e:
        print(f"  ✗ ERROR loading video info: {e}")
        return False
    
    # Step 2: Extract video frames and track what was extracted
    print("\n[2/5] Extracting video frames...")
    try:
        success, extracted_filenames = extract_video_frames(
            video_path, images_dir, frame_to_timestamp
        )
        if not success or not extracted_filenames:
            print("  ✗ ERROR: No frames were extracted")
            return False
        print(f"  ✓ Successfully extracted {len(extracted_filenames)} frames")
    except Exception as e:
        print(f"  ✗ ERROR extracting frames: {e}")
        return False
    
    # Step 3: Copy and process COLMAP files
    print("\n[3/5] Processing COLMAP files...")
    try:
        # Copy cameras.txt as-is
        shutil.copy(slam_dir / "cameras.txt", sparse_dir / "cameras.txt")
        print("  ✓ Copied cameras.txt")
        
        # Process images.txt with validation against extracted frames
        success = process_images_txt(
            slam_dir / "images.txt",
            sparse_dir / "images.txt",
            extracted_filenames
        )
        if not success:
            print("  ✗ ERROR: No valid images in images.txt")
            return False
        
        # Copy points3D.txt as-is
        shutil.copy(slam_dir / "points3D.txt", sparse_dir / "points3D.txt")
        print("  ✓ Copied points3D.txt")
    except Exception as e:
        print(f"  ✗ ERROR processing COLMAP files: {e}")
        return False
    
    # Step 4: Copy additional metadata (optional)
    print("\n[4/5] Copying additional metadata...")
    try:
        metadata_dir = output_root / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        if (inputs_dir / "gravity.txt").exists():
            shutil.copy(inputs_dir / "gravity.txt", metadata_dir / "gravity.txt")
            print("  ✓ Copied gravity.txt")
        
        if (inputs_dir / "traj_full.txt.bak").exists():
            shutil.copy(inputs_dir / "traj_full.txt.bak", metadata_dir / "trajectory.txt")
            print("  ✓ Copied trajectory.txt")
    except Exception as e:
        print(f"  ⚠ Warning: Could not copy metadata: {e}")
    
    # Step 5: Validate output
    print("\n[5/5] Validating output...")
    try:
        if not validate_output(images_dir, sparse_dir):
            print("  ✗ ERROR: Validation failed")
            return False
    except Exception as e:
        print(f"  ✗ ERROR during validation: {e}")
        return False
    
    print(f"\n✓ Successfully converted and validated: {dataset_name}")
    return True


def batch_convert(datasets_dir, output_base_dir, dataset_ids=None):
    """
    Convert multiple datasets in batch
    
    Args:
        datasets_dir: Base directory containing all datasets
        output_base_dir: Base output directory for processed datasets
        dataset_ids: List of specific dataset IDs to process (None = all)
    """
    datasets_dir = Path(datasets_dir)
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of datasets to process
    if dataset_ids:
        dataset_dirs = [datasets_dir / did for did in dataset_ids]
    else:
        dataset_dirs = sorted([d for d in datasets_dir.iterdir() if d.is_dir()])
    
    total = len(dataset_dirs)
    successful = 0
    failed = []
    
    print(f"\n{'#'*60}")
    print(f"# Batch Processing: {total} datasets")
    print(f"# Input:  {datasets_dir}")
    print(f"# Output: {output_base_dir}")
    print(f"{'#'*60}")
    
    for i, dataset_dir in enumerate(dataset_dirs, 1):
        dataset_name = dataset_dir.name
        output_dir = output_base_dir / dataset_name
        
        print(f"\n[{i}/{total}] Processing: {dataset_name}")
        
        try:
            success = convert_dataset(dataset_dir, output_dir)
            if success:
                successful += 1
            else:
                failed.append(dataset_name)
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed.append(dataset_name)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total datasets: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed datasets:")
        for name in failed:
            print(f"  - {name}")
    
    print(f"\nProcessed datasets are in: {output_base_dir}")
    print(f"\nTo train a dataset:")
    print(f"python train.py --source_path {output_base_dir}/<dataset_name> --model_path output/<dataset_name>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SLAM dataset(s) to 3DGS format with validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single dataset
  python convert_gs.py /path/to/Dataset/1748153841908 -o /path/to/processed_data/1748153841908
  
  # Batch convert all datasets
  python convert_gs.py /path/to/Dataset -o /path/to/processed_data --batch
  
  # Batch convert specific datasets
  python convert_gs.py /path/to/Dataset -o /path/to/processed_data --batch --ids 1748153841908 1748242779841
        """
    )
    
    parser.add_argument("dataset_path", type=str, 
                       help="Path to dataset directory or base datasets directory (for batch)")
    parser.add_argument("-o", "--output", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--batch", action="store_true",
                       help="Batch process all datasets in the input directory")
    parser.add_argument("--ids", nargs="+", type=str,
                       help="Specific dataset IDs to process (only with --batch)")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing
        batch_convert(args.dataset_path, args.output, args.ids)
    else:
        # Single dataset processing
        output_dir = Path(args.output)
        success = convert_dataset(args.dataset_path, output_dir)
        
        if success:
            print(f"\nTo train:")
            print(f"python train.py --source_path {output_dir} --model_path output/{Path(args.dataset_path).name}")
        else:
            print("\n✗ Conversion failed")
            exit(1)
