import os
import subprocess
import sys
import argparse

def run_metrics_for_all_scenes(output_base_dir, dataset_dir):
    # Check if directories exist
    if not os.path.exists(output_base_dir):
        print(f"Error: Output directory '{output_base_dir}' does not exist!")
        return
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist!")
        return
    
    # Find the most recent training run (timestamp directory)
    timestamp_dirs = [d for d in os.listdir(output_base_dir) 
                     if os.path.isdir(os.path.join(output_base_dir, d)) and d.count('_') == 1]
    
    if not timestamp_dirs:
        print("No training output directories found!")
        return
    
    latest_dir = max(timestamp_dirs)
    training_output_dir = os.path.join(output_base_dir, latest_dir)
    
    print(f"Computing metrics for models in: {training_output_dir}")
    
    # Find all scene directories in the dataset
    scene_dirs = [d for d in os.listdir(dataset_dir) 
                 if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for scene_name in scene_dirs:
        # Source path (original dataset)
        source_path = os.path.join(dataset_dir, scene_name)
        # Model path (trained output)  
        model_path = os.path.join(training_output_dir, scene_name)
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Skipping {scene_name}: Model not found at {model_path}")
            continue
            
        print(f"\nComputing metrics for: {scene_name}")
        
        cmd = [
            "python", "example_metrics.py",
            "-s", source_path,  # Source dataset path
            "-m", model_path,   # Model output path
            "--sh_degree", "3",
            "--source_type", "colmap"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error computing metrics for {scene_name}: {e}")
            print(e.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for all trained scenes")
    parser.add_argument("--output_dir", default="outputs", help="Output directory containing trained models")
    parser.add_argument("--dataset_dir", default="data/ProcessedDataset", help="Dataset directory with original scenes")
    
    args = parser.parse_args()
    run_metrics_for_all_scenes(args.output_dir, args.dataset_dir)