"""
Comprehensive debugging utilities for tracking Gaussian Splatting parameters and outputs
"""
import torch
import numpy as np
import os
from PIL import Image
import json


class GaussianDebugger:
    """Tracks and logs Gaussian parameters, rendered outputs, and training statistics"""

    def __init__(self, output_dir, enabled=True):
        self.enabled = enabled
        self.output_dir = output_dir
        self.debug_dir = os.path.join(output_dir, "debug")
        self.image_dir = os.path.join(self.debug_dir, "images")
        self.stats_file = os.path.join(self.debug_dir, "stats.jsonl")

        if self.enabled:
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs(self.image_dir, exist_ok=True)
            print(f"[DEBUG] Debug output directory: {self.debug_dir}")

    def log_pointcloud_init(self, init_xyz, init_color):
        """Log statistics about the initial point cloud from COLMAP"""
        if not self.enabled:
            return

        print("\n" + "="*80)
        print("INITIAL POINT CLOUD STATISTICS")
        print("="*80)

        if isinstance(init_xyz, torch.Tensor):
            xyz = init_xyz.detach().cpu().numpy()
            color = init_color.detach().cpu().numpy()
        else:
            xyz = init_xyz
            color = init_color

        print(f"Number of points: {xyz.shape[0]}")
        print(f"\nXYZ Statistics:")
        print(f"  Min:  [{xyz.min(axis=0)}]")
        print(f"  Max:  [{xyz.max(axis=0)}]")
        print(f"  Mean: [{xyz.mean(axis=0)}]")
        print(f"  Std:  [{xyz.std(axis=0)}]")

        print(f"\nColor Statistics (should be in [0, 1]):")
        print(f"  Min:  [{color.min(axis=0)}]")
        print(f"  Max:  [{color.max(axis=0)}]")
        print(f"  Mean: [{color.mean(axis=0)}]")
        print(f"  Std:  [{color.std(axis=0)}]")

        # Check for issues
        if color.min() < 0 or color.max() > 1:
            print(f"\n⚠️  WARNING: Colors are outside [0, 1] range!")
        if np.isnan(xyz).any() or np.isnan(color).any():
            print(f"\n⚠️  WARNING: NaN values detected in point cloud!")
        if np.isinf(xyz).any() or np.isinf(color).any():
            print(f"\n⚠️  WARNING: Inf values detected in point cloud!")

        print("="*80 + "\n")

    def log_gaussian_init(self, xyz, scale, rot, sh_0, sh_rest, opacity):
        """Log statistics about initialized Gaussian parameters"""
        if not self.enabled:
            return

        print("\n" + "="*80)
        print("INITIALIZED GAUSSIAN PARAMETERS")
        print("="*80)

        def tensor_stats(name, tensor):
            t = tensor.detach().cpu()
            print(f"\n{name}:")
            print(f"  Shape: {tensor.shape}")
            print(f"  Min:   {t.min().item():.6f}")
            print(f"  Max:   {t.max().item():.6f}")
            print(f"  Mean:  {t.mean().item():.6f}")
            print(f"  Std:   {t.std().item():.6f}")
            if torch.isnan(t).any():
                print(f"  ⚠️  WARNING: Contains NaN values!")
            if torch.isinf(t).any():
                print(f"  ⚠️  WARNING: Contains Inf values!")

        tensor_stats("XYZ (positions)", xyz)
        tensor_stats("Scale (log space)", scale)
        tensor_stats("Rotation (quaternions)", rot)
        tensor_stats("SH_0 (base color coefficients)", sh_0)
        tensor_stats("SH_rest (higher order SH)", sh_rest)
        tensor_stats("Opacity (logit space)", opacity)

        # Check activated values
        print(f"\n--- Activated Values (as used in rendering) ---")
        print(f"Scale (exp):    min={scale.exp().min().item():.6f}, max={scale.exp().max().item():.6f}, mean={scale.exp().mean().item():.6f}")
        print(f"Opacity (sigmoid): min={opacity.sigmoid().min().item():.6f}, max={opacity.sigmoid().max().item():.6f}, mean={opacity.sigmoid().mean().item():.6f}")

        if opacity.sigmoid().mean().item() < 0.01:
            print(f"⚠️  WARNING: Average opacity is very low ({opacity.sigmoid().mean().item():.6f})!")

        print("="*80 + "\n")

    def log_training_iteration(self, iteration, xyz, scale, rot, sh_0, sh_rest, opacity,
                               img, gt_image, loss, l1_loss, ssim_loss):
        """Log statistics during training"""
        if not self.enabled:
            return

        # Save statistics every N iterations
        if iteration % 100 == 0 or iteration < 5:
            stats = {
                "iteration": iteration,
                "loss": loss.item(),
                "l1_loss": l1_loss.item(),
                "ssim_loss": ssim_loss.item(),
                "gaussians": {
                    "count": xyz.shape[-1],
                    "opacity_mean": opacity.sigmoid().mean().item(),
                    "opacity_min": opacity.sigmoid().min().item(),
                    "opacity_max": opacity.sigmoid().max().item(),
                    "scale_mean": scale.exp().mean().item(),
                    "scale_min": scale.exp().min().item(),
                    "scale_max": scale.exp().max().item(),
                },
                "rendered_image": {
                    "min": img.min().item(),
                    "max": img.max().item(),
                    "mean": img.mean().item(),
                    "std": img.std().item(),
                },
                "gt_image": {
                    "min": gt_image.min().item(),
                    "max": gt_image.max().item(),
                    "mean": gt_image.mean().item(),
                }
            }

            # Append to stats file
            with open(self.stats_file, 'a') as f:
                f.write(json.dumps(stats) + '\n')

            # Print summary
            if iteration < 5 or iteration % 500 == 0:
                print(f"\n[Iter {iteration}] Loss: {loss.item():.6f} | "
                      f"Img range: [{img.min().item():.3f}, {img.max().item():.3f}] mean={img.mean().item():.3f} | "
                      f"Opacity: {opacity.sigmoid().mean().item():.3f}")

                # Check for issues
                if img.mean().item() < 0.01:
                    print(f"  ⚠️  WARNING: Rendered image is nearly black! (mean={img.mean().item():.6f})")
                if opacity.sigmoid().mean().item() < 0.01:
                    print(f"  ⚠️  WARNING: Gaussians have very low opacity! (mean={opacity.sigmoid().mean().item():.6f})")
                if torch.isnan(img).any():
                    print(f"  ⚠️  WARNING: NaN values in rendered image!")

    def save_comparison_image(self, iteration, rendered, gt, prefix="train"):
        """Save side-by-side comparison of rendered vs ground truth"""
        if not self.enabled:
            return

        # Only save periodically to avoid too many images
        if iteration % 100 != 0 and iteration not in [0, 1, 2, 3, 4]:
            return

        try:
            # Convert to numpy and ensure proper range
            if isinstance(rendered, torch.Tensor):
                rendered = rendered.detach().cpu().squeeze().permute(1, 2, 0).numpy()
            if isinstance(gt, torch.Tensor):
                gt = gt.detach().cpu().squeeze().permute(1, 2, 0).numpy()

            # Clip to valid range
            rendered = np.clip(rendered, 0, 1)
            gt = np.clip(gt, 0, 1)

            # Convert to uint8
            rendered_img = (rendered * 255).astype(np.uint8)
            gt_img = (gt * 255).astype(np.uint8)

            # Create side-by-side comparison
            h, w = rendered_img.shape[:2]
            comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
            comparison[:, :w] = rendered_img
            comparison[:, w:] = gt_img

            # Save
            output_path = os.path.join(self.image_dir, f"{prefix}_iter_{iteration:05d}.png")
            Image.fromarray(comparison).save(output_path)

            if iteration < 5:
                print(f"  [DEBUG] Saved comparison image: {output_path}")
        except Exception as e:
            print(f"  [DEBUG] Error saving comparison image: {e}")

    def check_render_output(self, img, iteration):
        """Check if rendered image has expected properties"""
        if not self.enabled:
            return

        img_min = img.min().item()
        img_max = img.max().item()
        img_mean = img.mean().item()

        issues = []
        if img_mean < 0.01:
            issues.append(f"Image is nearly black (mean={img_mean:.6f})")
        if img_min < -0.1:
            issues.append(f"Image has negative values (min={img_min:.6f})")
        if img_max > 1.1:
            issues.append(f"Image exceeds 1.0 (max={img_max:.6f})")
        if torch.isnan(img).any():
            issues.append("Image contains NaN values")
        if torch.isinf(img).any():
            issues.append("Image contains Inf values")

        if issues and (iteration < 10 or iteration % 500 == 0):
            print(f"\n  [Iter {iteration}] Render issues detected:")
            for issue in issues:
                print(f"    - {issue}")

    def save_final_summary(self, total_time, final_psnr=None):
        """Save final training summary"""
        if not self.enabled:
            return

        summary = {
            "total_training_time": total_time,
            "final_psnr": final_psnr,
        }

        summary_file = os.path.join(self.debug_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[DEBUG] Training summary saved to: {summary_file}")
        print(f"[DEBUG] Training took: {total_time:.2f}s")
        if final_psnr is not None:
            print(f"[DEBUG] Final PSNR: {final_psnr:.2f}")


# Global debugger instance
_global_debugger = None


def get_debugger():
    """Get the global debugger instance"""
    return _global_debugger


def init_debugger(output_dir, enabled=True):
    """Initialize the global debugger"""
    global _global_debugger
    _global_debugger = GaussianDebugger(output_dir, enabled)
    return _global_debugger
