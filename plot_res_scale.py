import os
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np

def plot_resolution_scales(args):
    """Plot resolution scales (reciprocal) for all scenes in different colors."""
    
    # Find output directory
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
            return
        output_dir = max(output_dirs)
        print(f"Using output directory: {output_dir}")
    
    # Find all scene directories
    scene_dirs = []
    for item in os.listdir(output_dir):
        full_path = os.path.join(output_dir, item)
        if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "point_cloud")):
            scene_dirs.append(full_path)
    
    scene_dirs = natsorted(scene_dirs)
    print(f"Found {len(scene_dirs)} scenes\n")
    
    # Setup plot with 13 distinct colors
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))  # Use tab20 for more colors
    
    scene_data = []
    
    # Load data from each scene
    for idx, scene_path in enumerate(scene_dirs):
        scene_name = os.path.basename(scene_path)
        point_cloud_dir = os.path.join(scene_path, "point_cloud")
        
        res_scale_json = None
        
        # Check finish directory first
        finish_res_scale = os.path.join(point_cloud_dir, "finish", "res_scale.json")
        if os.path.exists(finish_res_scale):
            res_scale_json = finish_res_scale
        else:
            # Check timeout directories
            timeout_dirs = [d for d in os.listdir(point_cloud_dir) if d.startswith("timeout_epoch_")]
            if timeout_dirs:
                timeout_dirs.sort(key=lambda x: int(x.split("_")[-1]), reverse=True)
                timeout_res_scale = os.path.join(point_cloud_dir, timeout_dirs[0], "res_scale.json")
                if os.path.exists(timeout_res_scale):
                    res_scale_json = timeout_res_scale
        
        if res_scale_json:
            with open(res_scale_json, 'r') as f:
                data = json.load(f)
            
            if data:
                global_steps = [entry["global_step"] for entry in data]
                # Plot reciprocal: 1 / render_scale
                render_scales_reciprocal = [1.0 / entry["render_scale"] for entry in data]
                
                # Shorten scene name for legend
                short_name = scene_name[:12] + "..." if len(scene_name) > 15 else scene_name
                
                plt.plot(global_steps, render_scales_reciprocal, 
                        color=colors[idx % 20], 
                        label=short_name,
                        linewidth=1.5, 
                        alpha=0.8)
                
                scene_data.append((scene_name, len(global_steps)))
                print(f"✓ {scene_name}: {len(global_steps)} steps")
            else:
                print(f"✗ {scene_name}: Empty res_scale.json")
        else:
            print(f"✗ {scene_name}: No res_scale.json found")
    
    if not scene_data:
        print("\nNo data to plot!")
        return
    
    # Configure plot
    plt.xlabel("Global Step", fontsize=12, fontweight='bold')
    plt.ylabel("Resolution (1 / Render Scale)", fontsize=12, fontweight='bold')
    plt.title("Resolution Schedule Across Scenes (Coarse → Fine)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Compact horizontal legend
    plt.legend(loc='upper center', 
              bbox_to_anchor=(0.5, -0.12),
              ncol=5,  # 5 columns for 13 scenes
              fontsize=8,
              framealpha=0.9,
              fancybox=True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "resolution_scales_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to {plot_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n--- Summary ---")
    print(f"Total scenes plotted: {len(scene_data)}")
    for scene_name, num_steps in scene_data:
        print(f"  {scene_name}: {num_steps} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot resolution scales for all scenes")
    parser.add_argument("--output_dir", default="outputs", 
                       help="Base output directory containing timestamped runs")
    parser.add_argument("--output_run_dir", default=None,
                       help="Specific run directory (overrides automatic detection)")
    
    args = parser.parse_args()
    plot_resolution_scales(args)