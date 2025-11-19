import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_training_config(log_file_path):
    """Extract iterations and position_lr_max_steps from log file."""
    iterations = None
    position_lr_max_steps = None
    
    with open(log_file_path, 'r') as f:
        content = f.read()
        
        # Search for --iterations
        iterations_match = re.search(r'--iterations\s+(\d+)', content)
        if iterations_match:
            iterations = int(iterations_match.group(1))
        
        # Search for --position_lr_max_steps
        lr_steps_match = re.search(r'--position_lr_max_steps\s+(\d+)', content)
        if lr_steps_match:
            position_lr_max_steps = int(lr_steps_match.group(1))
    
    return iterations, position_lr_max_steps


def create_combined_plot(folder_path):
    """Create combined bar graph for PSNR and time per scene."""
    
    folder_path = Path(folder_path)
    
    # Load metrics
    metrics_path = folder_path / "metrics.json"
    avg_metrics_path = folder_path / "average_metrics.json"
    
    if not metrics_path.exists():
        print(f"Error: metrics.json not found in {folder_path}")
        return
    
    if not avg_metrics_path.exists():
        print(f"Error: average_metrics.json not found in {folder_path}")
        return
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    with open(avg_metrics_path, 'r') as f:
        avg_metrics = json.load(f)
    
    # Find log file
    log_files = list(folder_path.glob("*.log"))
    if not log_files:
        print(f"Warning: No .log file found in {folder_path}")
        iterations, position_lr_max_steps = None, None
    else:
        log_file = log_files[0]
        iterations, position_lr_max_steps = extract_training_config(log_file)
    
    # Extract data
    scenes = list(metrics.keys())
    psnr_values = [metrics[scene]["PSNR"] for scene in scenes]
    time_values = [metrics[scene]["time"] for scene in scenes]
    avg_psnr = avg_metrics.get("average_PSNR", np.mean(psnr_values))
    
    # Create figure with two subplots (one for PSNR, one for time)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Scene indices for x-axis
    x = np.arange(len(scenes))
    bar_width = 0.6
    
    # Plot 1: PSNR
    bars1 = ax1.bar(x, psnr_values, bar_width, color='steelblue', alpha=0.8, label='PSNR')
    ax1.axhline(y=avg_psnr, color='red', linestyle='--', linewidth=2, label=f'Average PSNR: {avg_psnr:.2f}')
    ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(psnr_values) * 1.15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenes, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, psnr_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Training Time
    bars2 = ax2.bar(x, time_values, bar_width, color='darkorange', alpha=0.8, label='Training Time')
    ax2.set_xlabel('Scene ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(time_values) * 1.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenes, rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, time_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=8)
    
    # Overall title
    if iterations and position_lr_max_steps:
        title = f'Training Results per Scene\n(Iterations: {iterations}, Position LR Max Steps: {position_lr_max_steps})'
    else:
        title = 'Training Results per Scene'
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    output_path = folder_path / "training_results_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also display plot
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create combined bar graph for training results')
    parser.add_argument('--folder', type=str, 
                        default='outputs/20251117_140605',
                        help='Path to folder containing metrics.json and .log file')
    
    args = parser.parse_args()
    
    create_combined_plot(args.folder)