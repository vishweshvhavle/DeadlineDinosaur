import re
import sys
import matplotlib.pyplot as plt


def parse_log(path):
    """
    Parse the training log and extract epoch vs number of Gaussians (primitives)
    for each scene.

    We assume:
      - Scene header lines look like:
          [1/13] Training: 1747834320424 (using images_gt_downsampled)
      - Gaussians are reported in lines like:
          [DENSIFY][Epoch X] Gaussians: 68864 -> 68992 (Δ = +128), ...
        We'll track the final Gaussian count after each epoch.
    """

    # Scene header, e.g.:
    # [1/13] Training: 1747834320424 (using images_gt_downsampled)
    scene_header_re = re.compile(
        r"\[(\d+)/\d+\]\s+Training:\s+(\d+)"
    )

    # Training progress line to get epoch number, e.g.:
    # Training progress:  11%|█         | 5/46 [00:02<00:23,  1.76it/s]
    progress_re = re.compile(
        r"Training progress:.*?\|\s*(\d+)/(\d+)\s*\["
    )

    # Gaussians count from densification summary, e.g.:
    # [DENSIFY][Epoch 4] Gaussians: 68864 -> 68864 (Δ = +0), ...
    # [DENSIFY][Epoch 10] Gaussians: 68864 -> 68992 (Δ = +128), ...
    gaussians_re = re.compile(
        r"\[DENSIFY\]\[Epoch (\d+)\] Gaussians: \d+ -> (\d+)"
    )

    # Initial Gaussians count from training start, e.g.:
    # [TRAINING] Starting: 68839 primitives, 46 epochs
    initial_gaussians_re = re.compile(
        r"\[TRAINING\] Starting: (\d+) primitives"
    )

    scenes_data = {}
    current_scene_id = None
    current_scene_idx = None
    current_epoch = None
    last_gaussians = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Detect new scene
            m_scene = scene_header_re.search(line)
            if m_scene:
                current_scene_idx = int(m_scene.group(1))
                current_scene_id = m_scene.group(2)

                if current_scene_id not in scenes_data:
                    scenes_data[current_scene_id] = {
                        "index": current_scene_idx,
                        "epochs": [],
                        "gaussians": [],
                    }
                # Reset tracking for new scene
                current_epoch = None
                last_gaussians = None
                continue

            # Only process if we're in a scene block
            if current_scene_id is not None:
                # Look for initial Gaussians count
                m_init = initial_gaussians_re.search(line)
                if m_init and last_gaussians is None:
                    last_gaussians = int(m_init.group(1))
                    # Record epoch 0 with initial count
                    scenes_data[current_scene_id]["epochs"].append(0)
                    scenes_data[current_scene_id]["gaussians"].append(last_gaussians)

                # Look for training progress to get current epoch
                m_progress = progress_re.search(line)
                if m_progress:
                    current_epoch = int(m_progress.group(1))
                    # Only record if we have Gaussians data
                    if last_gaussians is not None and current_epoch > 0:
                        # We'll update this when we see the densification summary
                        pass

                # Look for Gaussians count in densification summary
                m_gauss = gaussians_re.search(line)
                if m_gauss:
                    epoch = int(m_gauss.group(1))
                    gaussians = int(m_gauss.group(2))
                    
                    # Update the Gaussians count
                    last_gaussians = gaussians
                    
                    # Make sure we have this epoch recorded
                    if current_scene_id in scenes_data:
                        data = scenes_data[current_scene_id]
                        if epoch not in data["epochs"]:
                            data["epochs"].append(epoch)
                            data["gaussians"].append(gaussians)
                        else:
                            # Update existing entry
                            idx = data["epochs"].index(epoch)
                            data["gaussians"][idx] = gaussians

                # Also capture the final completed epoch
                if "[TRAINING] Completed:" in line and last_gaussians is not None and current_epoch is not None:
                    if current_scene_id in scenes_data:
                        data = scenes_data[current_scene_id]
                        if current_epoch not in data["epochs"]:
                            data["epochs"].append(current_epoch)
                            data["gaussians"].append(last_gaussians)

    return scenes_data


def plot_scenes_to_png(
    scenes_data,
    out_path="gaussians_per_scene.png",
    title="Gaussians vs Epochs for All Scenes",
):
    """
    Plot epochs vs Gaussians for each scene and save to a PNG file.
    """
    # Filter scenes that actually have data
    non_empty_scenes = {
        sid: data
        for sid, data in scenes_data.items()
        if data["epochs"] and data["gaussians"]
    }

    if not non_empty_scenes:
        print("No epoch/Gaussians data found for any scene.")
        return

    plt.figure(figsize=(10, 6))

    # Sort scenes by their numeric index [k/13] so legend is ordered
    sorted_scenes = sorted(
        non_empty_scenes.items(),
        key=lambda kv: kv[1]["index"],
    )

    for scene_id, data in sorted_scenes:
        epochs = data["epochs"]
        gaussians = data["gaussians"]

        label = f"Scene {data['index']} ({scene_id})"
        plt.plot(
            epochs,
            gaussians,
            marker="o",
            linewidth=1,
            markersize=3,
            label=label,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Number of Gaussians")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_num_gaussians.py <log_file> [output_png]")
        sys.exit(1)

    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else "gaussians_per_scene.png"

    scenes_data = parse_log(log_path)

    if not scenes_data:
        print("No scene headers found in the log.")
        sys.exit(1)

    # Print debug info
    for scene_id, data in scenes_data.items():
        print(f"Scene {data['index']} ({scene_id}): {len(data['epochs'])} data points")
        if data['epochs']:
            print(f"  Epochs: {data['epochs']}")
            print(f"  Gaussians: {data['gaussians']}")

    plot_scenes_to_png(scenes_data, out_path=out_path)


if __name__ == "__main__":
    main()