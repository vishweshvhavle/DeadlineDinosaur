import os
import sys
import argparse
from datetime import datetime
from natsort import natsorted
from tqdm import tqdm
import subprocess
import tempfile

def run_pipeline(args):
    # Find scene folders
    scene_folders = []
    for item in os.listdir(args.dataset_dir):
        full_path = os.path.join(args.dataset_dir, item)
        if os.path.isdir(full_path) and \
           os.path.isdir(os.path.join(full_path, 'images')) and \
           os.path.isdir(os.path.join(full_path, 'sparse')):
            scene_folders.append(full_path)
    
    if not scene_folders:
        print(f"No valid scenes in {args.dataset_dir}")
        sys.exit(1)

    scene_folders = natsorted(scene_folders)
    print(f"Found {len(scene_folders)} scenes")

    # Create datetime-based output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Training config
    train_config = (
        "--sh_degree 3 --source_type colmap "
        "--target_primitives 1000000 --iterations 30000 "
        "--position_lr_max_steps 4500 --position_lr_final 0.000016 "
        "--densification_interval 2"
    )

    # Train each scene
    for i, source_path in enumerate(scene_folders):
        scene_name = os.path.basename(source_path)
        model_path = os.path.join(output_dir, scene_name)

        print(f"\n[{i+1}/{len(scene_folders)}] Training: {scene_name}")

        debug_flag = "--debug" if args.debug else ""
        train_cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu} CUDA_LAUNCH_BLOCKING=1 python example_train.py "
            f"-s {source_path} -m {model_path} {train_config} {debug_flag}"
        )
        print(train_cmd)
        os.system(train_cmd)

    # Compute metrics
    print("\n--- Computing metrics ---")
    results = []
    metrics_config = "--sh_degree 3 --source_type colmap"
    
    for source_path in scene_folders:
        scene_name = os.path.basename(source_path)
        model_path = os.path.join(output_dir, scene_name)

        print(f"\n[Metrics] Processing: {scene_name}")
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp:
            tmp_path = tmp.name
        
        cmd = f"python example_metrics.py -s {source_path} -m {model_path} {metrics_config} | tee {tmp_path}"
        
        # Run command - tqdm will display, output will be saved to temp file
        result = os.system(cmd)
        
        if result != 0:
            print(f"Error computing metrics for {scene_name}")
            os.unlink(tmp_path)
            continue

        # Read PSNR from temp file
        with open(tmp_path, 'r') as f:
            output = f.read()
        
        os.unlink(tmp_path)
        
        idx = output.find('  PSNR : ')
        if idx != -1:
            end = output[idx+9:].find('\n')
            psnr = float(output[idx+9:idx+9+end])
            results.append(psnr)
            print(f"  → {scene_name}: PSNR = {psnr:.2f}")
        else:
            print(f"  → Could not parse PSNR for {scene_name}")
        
    if results:
        import numpy as np
        print(f"\nAverage PSNR: {np.mean(results):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/ProcessedDataset")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save visualization every 5 seconds")

    args = parser.parse_args()
    run_pipeline(args)