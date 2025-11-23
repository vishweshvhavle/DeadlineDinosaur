import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

def parse_evaluation_log(log_path):
    if not Path(log_path).exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    scene_ids = []
    psnrs = []
    pattern = re.compile(r"→\s*(\d+)\s*:\sPSNR\s=\s*([0-9.]+)")
    
    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                scene_ids.append(match.group(1))
                psnrs.append(float(match.group(2)))
    
    if not scene_ids:
        print("No evaluation lines found in log!")
        return None
    
    return dict(zip(scene_ids, psnrs))

def plot_psnr(psnr_dict, output_png="psnr_plot.png"):
    psnr_data = sorted(psnr_dict.items(), key=lambda x: x[0])
    scene_ids = [scene for scene, _ in psnr_data]
    psnr_values = [float(psnr) for _, psnr in psnr_data]
    
    mean_psnr = sum(psnr_values) / len(psnr_values)
    
    fig, ax = plt.subplots(figsize=(16, 4))
    
    x_pos = range(len(scene_ids))
    bars = ax.bar(x_pos, psnr_values, color="#F5B680", edgecolor='none')
    
    # Scene IDs at bottom with offset
    for i, (rect, scene_id) in enumerate(zip(bars, scene_ids)):
        ax.text(i, 3, scene_id, ha='center', va='bottom', 
                fontsize=12, rotation=90, color='black')
    
    # PSNR values on top
    for i, (rect, psnr) in enumerate(zip(bars, psnr_values)):
        ax.text(i, psnr + 0.5, f"{psnr:.2f}", ha='center', va='bottom', fontsize=12)
    
    # Mean line
    ax.axhline(mean_psnr, color='red', linestyle='--', linewidth=1.5, 
               label=f'Mean: {mean_psnr:.2f} dB')
    
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_xticks([])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, max(psnr_values) + 5)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_png}")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, required=True)
    args = parser.parse_args()
    
    psnr_dict = parse_evaluation_log(args.log_file)
    if psnr_dict:
        print(f"Parsed {len(psnr_dict)} scenes")
        plot_psnr(psnr_dict)