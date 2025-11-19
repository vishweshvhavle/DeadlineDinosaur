import os
import json
import argparse

if __name__ == "__main__":
    # Take input argument as path to outputs dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, required=True, help="Path to the outputs dir containing training logs)")
    args = parser.parse_args()
    
    outputs_dir = args.outputs_dir

    total_training_time = 0.0
    for scene in os.listdir(outputs_dir):
        scene_dir_pc = os.path.join(outputs_dir, scene, "point_cloud")

        for folder in os.listdir(scene_dir_pc):
            temp_dir = os.path.join(scene_dir_pc, folder)
            break
        
        json_path = os.path.join(temp_dir, "training_metrics.json")

        # open json file
        with open(json_path, "r") as f:
            metrics = json.load(f)
            training_time = metrics["training_time_seconds"]

        print(f"Scene: {scene}, Training Time (seconds): {training_time}")
        total_training_time += training_time
    
    avg_training_time = total_training_time / len(os.listdir(outputs_dir))
    print(f"Average Training Time (seconds): {avg_training_time}")

# python compute_avg_time.py --outputs_dir outputs/20251115_215015