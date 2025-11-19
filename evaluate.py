import os
import sys
import json
import argparse
from datetime import datetime
from natsort import natsorted
from tqdm import tqdm
import tempfile
import numpy as np

def evaluate(args):
    # Find scene folders
    scene_folders = []
    scene_image_dirs = {}  # Track which image directory each scene uses
    for item in os.listdir(args.dataset_dir):
        full_path = os.path.join(args.dataset_dir, item)
        if not os.path.isdir(full_path):
            continue

        # Check for sparse directory
        if not os.path.isdir(os.path.join(full_path, 'sparse')):
            continue

        # Check for either images_gt_downsampled or images
        if os.path.isdir(os.path.join(full_path, 'images_gt_downsampled')):
            scene_folders.append(full_path)
            scene_image_dirs[full_path] = 'images_gt_downsampled'
        elif os.path.isdir(os.path.join(full_path, 'images')):
            scene_folders.append(full_path)
            scene_image_dirs[full_path] = 'images'
    
    if not scene_folders:
        print(f"No valid scenes in {args.dataset_dir}")
        sys.exit(1)
    scene_folders = natsorted(scene_folders)
    print(f"Found {len(scene_folders)} scenes")
    
    if args.output_run_dir:
        output_dir = args.output_run_dir
        print(f"Using specified output directory: {output_dir}")
    else:
        output_dirs = []
        for item in os.listdir(args.output_dir):
            full_path = os.path.join(args.output_dir, item)
            if os.path.isdir(full_path):
                try:
                    datetime.strptime(item, "%Y%m%d_%H%M%S")
                    output_dirs.append(full_path)
                except ValueError:
                    continue
        if not output_dirs:
            print(f"No valid output directories in {args.output_dir}")
            sys.exit(1)
        latest_output_dir = max(output_dirs)
        output_dir = latest_output_dir
        print(f"Using output directory: {output_dir}")
    
    print("\n--- Computing metrics ---")
    results = []
    times = []
    all_metrics = {}
    metrics_config_base = "--sh_degree 3 --source_type colmap"

    for source_path in scene_folders:
        scene_name = os.path.basename(source_path)
        model_path = os.path.join(output_dir, scene_name)
        image_dir = scene_image_dirs[source_path]
        print(f"\n[Metrics] Processing: {scene_name} (using {image_dir})")

        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp:
            tmp_path = tmp.name

        cmd = f"python example_metrics.py -s {source_path} -m {model_path} {metrics_config_base} --images {image_dir} | tee {tmp_path}"
        result = os.system(cmd)
        
        if result != 0:
            print(f"Error computing metrics for {scene_name}")
            os.unlink(tmp_path)
            continue
        
        with open(tmp_path, 'r') as f:
            output = f.read()
        os.unlink(tmp_path)
        
        idx = output.find('  PSNR : ')
        if idx != -1:
            end = output[idx+9:].find('\n')
            psnr = float(output[idx+9:idx+9+end])
            results.append(psnr)
            print(f"  → {scene_name}: PSNR = {psnr:.2f}")
            
            # --- MODIFIED: Load training time from finish or timeout directories ---
            time_seconds = None
            point_cloud_dir = os.path.join(model_path, "point_cloud")
            if os.path.exists(point_cloud_dir):
                # First check the 'finish' directory (highest priority)
                finish_json = os.path.join(point_cloud_dir, "finish", "training_metrics.json")
                if os.path.exists(finish_json):
                    with open(finish_json) as f:
                        train_data = json.load(f)
                        time_seconds = train_data.get("time", None)
                        if time_seconds is not None:
                            times.append(time_seconds)
                            print(f"  → Loaded time from finish: {time_seconds:.2f}s")
                else:
                    # Fall back to timeout directories
                    timeout_dirs = [d for d in os.listdir(point_cloud_dir) if d.startswith("timeout_epoch_")]
                    if timeout_dirs:
                        # Use the timeout checkpoint with the highest epoch number
                        timeout_dirs.sort(key=lambda x: int(x.split("_")[-1]), reverse=True)
                        training_json = os.path.join(point_cloud_dir, timeout_dirs[0], "training_metrics.json")
                        if os.path.exists(training_json):
                            with open(training_json) as f:
                                train_data = json.load(f)
                                time_seconds = train_data.get("time", None)
                                if time_seconds is not None:
                                    times.append(time_seconds)
                                    print(f"  → Loaded time from timeout: {time_seconds:.2f}s")
            # --- END MODIFIED ---
            
            # Save metrics.json
            metrics_data = {
                "PSNR": round(psnr, 2),
                "time": time_seconds
            }
            all_metrics[scene_name] = metrics_data
            metrics_path = os.path.join(model_path, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            print(f"  → Saved {metrics_path}")
        else:
            print(f"  → Could not parse PSNR for {scene_name}")
    
    if results:
        avg_psnr = np.mean(results)
        avg_time = np.mean(times) if times else None
        print(f"\nAverage PSNR: {avg_psnr:.2f}")
        if avg_time is not None:
            print(f"Average Time: {avg_time:.2f}s")
        
        # Save average_metrics.json
        average_metrics = {
            "average_PSNR": round(avg_psnr, 2),
            "average_time": round(avg_time, 2) if avg_time is not None else None,
            "num_scenes": len(results)
        }
        avg_metrics_path = os.path.join(output_dir, "average_metrics.json")
        with open(avg_metrics_path, 'w') as f:
            json.dump(average_metrics, f, indent=2)
        print(f"\nSaved {avg_metrics_path}")
        
        # Save combined metrics.json
        combined_metrics_path = os.path.join(output_dir, "metrics.json")
        with open(combined_metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Saved {combined_metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/eval_data_pinhole")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--output_run_dir") 
    parser.add_argument("--gpu", default="0")
    
    args = parser.parse_args()
    evaluate(args)