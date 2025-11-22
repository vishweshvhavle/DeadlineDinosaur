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
          Training progress:  10%|█         | 4/40 [...][MOMENTUM] ... current=206517, P_fin: ...
        where '4' is the epoch index and 'current' is the total number of primitives.
    """

    # Scene header, e.g.:
    # [1/13] Training: 1747834320424 (using images_gt_downsampled)
    scene_header_re = re.compile(
        r"\[(\d+)/\d+\]\s+Training:\s+(\d+)"
    )

    # Training progress + MOMENTUM + current, e.g.:
    # Training progress:  10%|█         | 4/40 [00:02<...][MOMENTUM] ... current=206517, P_fin: ...
    momentum_re = re.compile(
        r"Training progress:.*?\|\s*(\d+)/(\d+)\s*\[.*?MOMENTUM.*?current=(\d+)",
        re.DOTALL,
    )

    scenes_data = {}
    current_scene_id = None
    current_scene_idx = None

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
                continue

            # Only look for MOMENTUM/current lines inside a scene block
            if current_scene_id is not None:
                m_mom = momentum_re.search(line)
                if m_mom:
                    # epoch_idx / total_epochs
                    epoch = int(m_mom.group(1))
                    # total_epochs = int(m_mom.group(2))  # Not used, but available
                    current_gaussians = int(m_mom.group(3))

                    scenes_data[current_scene_id]["epochs"].append(epoch)
                    scenes_data[current_scene_id]["gaussians"].append(current_gaussians)

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
        if data["epochs"]
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
    plt.ylabel("Number of Gaussians (current)")
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

    plot_scenes_to_png(scenes_data, out_path=out_path)


if __name__ == "__main__":
    main()
