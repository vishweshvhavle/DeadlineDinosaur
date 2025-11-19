import os
import sys
import argparse
from datetime import datetime
from natsort import natsorted
from tqdm import tqdm

def train(args):
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

    # Create datetime-based output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Base training config (without --images which varies per scene)
    train_config_base = (
        "--sh_degree 3 --source_type colmap "
        "--target_primitives 1000000 --iterations 10000 "
        "--position_lr_max_steps 10000 --position_lr_final 0.000016 "
        "--densification_interval 2 "
        "--resolution_mode freq"
    )

    # Train each scene
    for i, source_path in enumerate(scene_folders):
        scene_name = os.path.basename(source_path)
        model_path = os.path.join(output_dir, scene_name)
        image_dir = scene_image_dirs[source_path]

        print(f"\n[{i+1}/{len(scene_folders)}] Training: {scene_name} (using {image_dir})")

        train_cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu} CUDA_LAUNCH_BLOCKING=1 python example_train.py "
            f"-s {source_path} -m {model_path} {train_config_base} --images {image_dir}"
        )
        print(train_cmd)
        os.system(train_cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/eval_data_pinhole")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--gpu", default="0")
    
    args = parser.parse_args()
    train(args)