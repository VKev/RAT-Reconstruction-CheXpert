#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Optional

import torch
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from PIL import Image
import numpy as np

from datamodule import ImageDataModule
from utils import save_mask_tensor, rle_encode_binary, rle_decode_binary


def build_sam_predictor(checkpoint: str, model_type: str = "vit_b", device: str | None = None):
    from segment_anything import sam_model_registry, SamPredictor
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor


@torch.no_grad()
def predict_mask_for_image(predictor, image_path: str, target_size: int = 224) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    # Resize to target_size x target_size to match training transforms (e.g., 224x224)
    try:
        resample = Image.BILINEAR
        image = image.resize((target_size, target_size), resample)
    except Exception:
        image = image.resize((target_size, target_size))

    image_np = np.array(image)
    predictor.set_image(image_np)
    
    # Generate automatic masks using SAM's automatic mask generation
    from segment_anything import SamAutomaticMaskGenerator
    mask_generator = SamAutomaticMaskGenerator(predictor.model)
    masks = mask_generator.generate(image_np)
    
    # Combine ALL masks into one multi-class mask (each region gets different ID)
    if masks:
        h, w = image_np.shape[:2]
        combined_mask = np.zeros((h, w), dtype="int64")
        
        # Sort masks by area (largest first) to prioritize bigger segments
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Assign different IDs to different masks
        for i, mask_data in enumerate(masks_sorted):
            mask_region = mask_data['segmentation'].astype(bool)
            # Only assign to pixels not already assigned (background = 0)
            combined_mask[mask_region & (combined_mask == 0)] = i + 1
        
        mask_np = combined_mask
    else:
        # Fallback: create grid mask with 14x14 patches
        h, w = image_np.shape[:2]
        grid_h, grid_w = 14, 14
        
        # Calculate patch size
        patch_h = h // grid_h
        patch_w = w // grid_w
        
        mask_np = np.zeros((h, w), dtype="int64")
        region_id = 1
        
        for i in range(grid_h):
            for j in range(grid_w):
                start_h = i * patch_h
                end_h = min((i + 1) * patch_h, h)
                start_w = j * patch_w
                end_w = min((j + 1) * patch_w, w)
                
                mask_np[start_h:end_h, start_w:end_w] = region_id
                region_id += 1
        
        print(f"[SAM] No masks detected, using 14x14 grid fallback ({grid_h * grid_w} regions)")
    
    mask = torch.from_numpy(mask_np)
    return mask


def main():
    parser = argparse.ArgumentParser(description="Generate offline masks for CheXpert using SAM")
    parser.add_argument("--dataset", type=str, default="chexpert", choices=["chexpert"], help="Only chexpert supported")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save .pt masks (or single file with --single-file)")
    parser.add_argument("--single-file", action="store_true", help="Save all masks into one single JSONL file (RLE)")
    parser.add_argument("--jsonl", action="store_true", help="Force JSONL output for per-image saves (RLE)")
    parser.add_argument("--sam-checkpoint", type=str, required=True, help="Path to SAM checkpoint (.pth)")
    parser.add_argument("--sam-model-type", type=str, default="vit_b", choices=["vit_b","vit_l","vit_h"], help="SAM model type")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"], help="Device for SAM inference")
    parser.add_argument("--target-size", type=int, default=224, help="Resize input image to this square size before SAM (e.g., 224)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for datamodule/dataloader (SAM inference remains per-image)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers for datamodule")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel SAM workers per GPU (duplicates model per worker; requires free VRAM)")
    # CheXpert paths
    parser.add_argument("--chexpert-train-csv", type=str, default=r"C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\data\archive\train.csv",
                        help="Path to CheXpert training CSV (with 'Path' column)")
    parser.add_argument("--chexpert-val-csv", type=str, default=r"C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\data\archive\valid.csv",
                        help="Path to CheXpert validation CSV (with 'Path' column)")
    parser.add_argument("--chexpert-test-csv", type=str, default=r"C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\data\ChetXpert_Test\content\chexlocalize\chexlocalize\CheXpert\test_labels.csv",
                        help="Path to CheXpert test CSV (with 'Path' column)")
    parser.add_argument("--chexpert-root", type=str, default=None,
                        help="Root directory to resolve image paths from the CSV (optional if CSV has absolute paths)")
    parser.add_argument("--chexpert-train-root", type=str, default=r"C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\data\archive",
                        help="Override root for train split (joined as <root>/train/<suffix> if needed)")
    parser.add_argument("--chexpert-valid-root", type=str, default=r"C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\data\archive",
                        help="Override root for valid split (joined as <root>/valid/<suffix> if needed)")
    parser.add_argument("--chexpert-test-root", type=str, default=r"C:\Vkev\Repos\Region-Attention-Transformer-for-Medical-Image-Restoration\data\ChetXpert_Test\content\chexlocalize\chexlocalize\CheXpert",
                        help="Override root for test split (joined as <root>/test/<suffix> if needed)")
    parser.add_argument("--chexpert-policy", type=str, default="ones", choices=["ones", "zeroes"],
                        help="How to map uncertain labels (-1): 'ones' or 'zeroes'")
    parser.add_argument("--only-support-devices", action="store_true", help="Generate only for Support Devices samples")
    parser.add_argument("--exclude-support-devices", action="store_true")
    # Limits
    parser.add_argument("--limit-train-fraction", type=float, default=1.0, help="Use only a fraction of train set [0,1]")
    parser.add_argument("--limit-val-fraction", type=float, default=1.0, help="Use only a fraction of val set [0,1]")
    parser.add_argument("--limit-test-fraction", type=float, default=1.0, help="Use only a fraction of test set [0,1]")
    # Small visualization after save
    parser.add_argument("--viz", action="store_true", help="Visualize a few saved masks over images")
    parser.add_argument("--viz-num-samples", type=int, default=2, help="Number of samples to visualize")
    parser.add_argument("--viz-out-dir", type=str, default="outputs/masks_viz", help="Directory to save visualization images")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    use_device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and use_device != "cuda":
        print("[gen_masks] CUDA requested but not available. Falling back to CPU.")
    predictor = build_sam_predictor(args.sam_checkpoint, args.sam_model_type, device=use_device)

    # Resume/autosave state for single-file JSONL
    import json
    single_file_path = os.path.join(args.output_dir, "chexpert_masks.jsonl") if args.single_file else None
    existing_paths = set()
    if args.single_file and os.path.exists(single_file_path):
        try:
            with open(single_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        p = rec.get("path")
                        if isinstance(p, str):
                            existing_paths.add(p)
                    except Exception:
                        continue
            print(f"[resume] Loaded {len(existing_paths)} existing entries from {single_file_path}")
        except Exception as e:
            print(f"[resume] Failed to load existing JSONL: {e}")
    pending_records = []  # records queued for append (sequential path)

    def flush_pending():
        if not args.single_file or not pending_records:
            return
        os.makedirs(args.output_dir, exist_ok=True)
        try:
            with open(single_file_path, "a", encoding="utf-8") as f:
                for rec in pending_records:
                    f.write(json.dumps(rec) + "\n")
            print(f"[autosave] Appended {len(pending_records)} records to {single_file_path}")
        except Exception as e:
            print(f"[autosave] Failed to append records: {e}")
        finally:
            pending_records.clear()

    dm = ImageDataModule(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chexpert_train_csv=args.chexpert_train_csv,
        chexpert_val_csv=args.chexpert_val_csv,
        chexpert_test_csv=args.chexpert_test_csv,
        chexpert_root=args.chexpert_root,
        chexpert_train_root=args.chexpert_train_root,
        chexpert_valid_root=args.chexpert_valid_root,
        chexpert_test_root=args.chexpert_test_root,
        chexpert_policy=args.chexpert_policy,
        chexpert_only_support_devices=args.only_support_devices,
        chexpert_exclude_support_devices=args.exclude_support_devices,
    )
    dm.prepare_data()

    all_masks = [] if (args.single_file and False) else None  # deprecated collector when resume is on
    viz_candidates = []  # list of tuples: (norm_rel_path, split, abs_path)

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    writer_lock = threading.Lock()
    viz_lock = threading.Lock()
    thread_local = threading.local()
    use_threads = max(1, int(args.concurrency)) > 1 and args.device == "cuda"

    def get_thread_predictor():
        p = getattr(thread_local, "predictor", None)
        if p is None:
            p = build_sam_predictor(args.sam_checkpoint, args.sam_model_type, device=use_device)
            thread_local.predictor = p
        return p

    try:
        for split in ["fit", "validate", "test"]:
            try:
                # Clear GPU cache before switching datasets
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                dm.setup(split if split != "fit" else "fit")
                if split == "fit":
                    dataset = dm.train_set
                elif split == "validate":
                    dataset = dm.val_set
                else:
                    dataset = dm.test_set
            except Exception:
                continue

            total = len(dataset)
            if split == "fit":
                frac = max(0.0, min(1.0, float(args.limit_train_fraction)))
            elif split == "validate":
                frac = max(0.0, min(1.0, float(args.limit_val_fraction)))
            else:
                frac = max(0.0, min(1.0, float(args.limit_test_fraction)))
            limit_n = max(0, int(total * frac)) if frac < 1.0 else total
            if limit_n == 0:
                continue

            interval = max(1, int(limit_n * 0.1))  # autosave every 10% (sequential)

            def build_norm_path(rel_path_str: str) -> str:
                n = rel_path_str.replace("\\", "/")
                found = False
                for marker in ["train/", "valid/", "test/"]:
                    pos = n.find(marker)
                    if pos != -1:
                        n = n[pos:]
                        found = True
                        break
                if not found:
                    if split == "fit":
                        n = "train/" + n
                    elif split == "validate":
                        n = "valid/" + n
                    else:
                        n = "test/" + n
                return n

            def process_one(index: int, use_thread_predictor: bool) -> None:
                try:
                    rel = dataset.rel_paths[index]
                    abs_p = dataset._resolve_path(rel)
                    norm_p = build_norm_path(rel)

                    # Resume skipping logic
                    if args.single_file:
                        # quick check without lock
                        if norm_p in existing_paths:
                            return
                    else:
                        if args.jsonl:
                            op = os.path.join(args.output_dir, os.path.splitext(norm_p)[0] + ".jsonl")
                            if os.path.exists(op):
                                return
                        else:
                            op = os.path.join(args.output_dir, os.path.splitext(norm_p)[0] + ".pt")
                            if os.path.exists(op):
                                return

                    # get predictor
                    pred_inst = get_thread_predictor() if use_thread_predictor else predictor
                    m = predict_mask_for_image(pred_inst, abs_p, target_size=args.target_size)

                    if args.single_file:
                        m_np = m.cpu().numpy().astype("uint16")
                        rec = {"path": norm_p, "mask": m_np.tolist()}
                        if use_threads:
                            with writer_lock:
                                os.makedirs(args.output_dir, exist_ok=True)
                                with open(single_file_path, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(rec) + "\n")
                                existing_paths.add(norm_p)
                        else:
                            pending_records.append(rec)
                            existing_paths.add(norm_p)
                    else:
                        if args.jsonl:
                            m_bin = (m > 0).to(torch.uint8).cpu().numpy()
                            rec = {"path": norm_p, "rle": rle_encode_binary(m_bin)}
                            op = os.path.join(args.output_dir, os.path.splitext(norm_p)[0] + ".jsonl")
                            os.makedirs(os.path.dirname(op), exist_ok=True)
                            with open(op, "a", encoding="utf-8") as f:
                                f.write(json.dumps(rec) + "\n")
                        else:
                            op = os.path.join(args.output_dir, os.path.splitext(norm_p)[0] + ".pt")
                            save_mask_tensor(m, op)

                    # collect viz candidates
                    if args.viz:
                        with viz_lock:
                            if len(viz_candidates) < max(1, int(args.viz_num_samples)):
                                viz_candidates.append((norm_p, split, abs_p))
                except Exception:
                    return

            if use_threads:
                with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as executor:
                    futures = [executor.submit(process_one, idx, True) for idx in range(limit_n)]
                    for _ in tqdm(as_completed(futures), total=limit_n, desc=f"SAM masks {split} ({limit_n}/{total})"):
                        pass
            else:
                for idx in tqdm(range(limit_n), desc=f"SAM masks {split} ({limit_n}/{total})"):
                    process_one(idx, False)
                    # autosave by 10%
                    if args.single_file and ((idx + 1) % interval) == 0:
                        flush_pending()

            # flush end of split (sequential path)
            if not use_threads:
                flush_pending()
    except KeyboardInterrupt:
        # ensure pending data are saved
        flush_pending()
        print("[resume] Caught KeyboardInterrupt. Saved pending records. Exiting early.")
        return

    if args.single_file:
        # Ensure any remaining records are flushed
        flush_pending()
        if single_file_path and os.path.exists(single_file_path):
            print(f"[gen_masks] Saved masks to {single_file_path}")
        
        # Visualize a few samples from the saved JSONL to verify multi-class masks
        if args.viz and len(all_masks) > 0:
            print(f"[gen_masks] Creating sample visualizations from saved JSONL...")
            os.makedirs(args.viz_out_dir + "_jsonl_test", exist_ok=True)
            
            # Take first 2 samples for quick verification
            for idx, sample in enumerate(all_masks[:2]):
                try:
                    # Find corresponding image path
                    rel_path = sample["path"]
                    
                    # Resolve absolute path robustly from normalized rel_path
                    def _resolve_abs_from_norm(norm_path: str) -> str:
                        norm_path = norm_path.replace("\\", "/")
                        if norm_path.startswith("train/"):
                            suffix = norm_path[len("train/"):]
                            return os.path.normpath(os.path.join(args.chexpert_train_root, "train", suffix))
                        if norm_path.startswith("valid/"):
                            suffix = norm_path[len("valid/"):]
                            return os.path.normpath(os.path.join(args.chexpert_valid_root, "valid", suffix))
                        if norm_path.startswith("test/"):
                            suffix = norm_path[len("test/"):]
                            return os.path.normpath(os.path.join(args.chexpert_test_root, "test", suffix))
                        return norm_path

                    abs_path = _resolve_abs_from_norm(rel_path)
                    
                    if os.path.exists(abs_path):
                        # Get mask from saved data
                        mask_data = sample.get("mask", None)
                        if mask_data:
                            mask_np = np.array(mask_data, dtype="int64")
                            
                            # Create visualization
                            safe_name = rel_path.replace("/", "__").replace("\\", "__")
                            save_prefix = os.path.join(args.viz_out_dir + "_jsonl_test", f"jsonl_test_{idx+1}_{safe_name}")
                            
                            # Create simple visualization function for JSONL test
                            def _overlay_and_save_simple(img_path: str, mask_np, save_prefix: str):
                                from PIL import Image
                                import matplotlib.pyplot as plt
                                
                                img = Image.open(img_path).convert("RGB")
                                img = img.resize((args.target_size, args.target_size))
                                img_np = np.array(img)
                                unique_ids = np.unique(mask_np)
                                num_classes = len(unique_ids) - 1 if 0 in unique_ids else len(unique_ids)
                                
                                # Quick comparison visualization
                                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                                
                                axes[0].imshow(img_np)
                                axes[0].set_title('Original')
                                axes[0].axis('off')
                                
                                im = axes[1].imshow(mask_np, cmap='tab20', vmin=0, vmax=20)
                                axes[1].set_title(f'Mask ({num_classes} regions)')
                                axes[1].axis('off')
                                
                                axes[2].imshow(img_np)
                                masked = np.ma.masked_where(mask_np == 0, mask_np)
                                axes[2].imshow(masked, alpha=0.5, cmap='tab20', vmin=0, vmax=20)
                                axes[2].set_title('Overlay')
                                axes[2].axis('off')
                                
                                plt.colorbar(im, ax=axes[1], shrink=0.6)
                                plt.tight_layout()
                                plt.savefig(save_prefix + "_jsonl_verification.png", bbox_inches='tight', dpi=150)
                                plt.close()
                                
                                print(f"[jsonl_test] Verified mask with {num_classes} regions: {save_prefix}_jsonl_verification.png")
                            
                            _overlay_and_save_simple(abs_path, mask_np, save_prefix)
                            
                except Exception as e:
                    print(f"[gen_masks] Failed to visualize sample {idx}: {e}")
                    continue
                else:
                    if not os.path.exists(abs_path):
                        print(f"[jsonl_test] Image not found, skip visualization: {abs_path}")

    # Optional small visualization (draw overlay using saved masks)
    if args.viz and viz_candidates:
        os.makedirs(args.viz_out_dir, exist_ok=True)

        def _overlay_and_save(img_path: str, mask_np, save_prefix: str):
            from PIL import Image
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            img = Image.open(img_path).convert("RGB")
            img = img.resize((args.target_size, args.target_size))
            img_np = np.array(img)
            
            # Handle multi-class mask visualization
            unique_ids = np.unique(mask_np)
            num_classes = len(unique_ids) - 1 if 0 in unique_ids else len(unique_ids)
            
            # Method 1: Multi-colored overlay (each segment different color)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(img_np)
            # Use a colormap that gives distinct colors
            masked = np.ma.masked_where(mask_np == 0, mask_np)
            ax.imshow(masked, alpha=0.6, cmap='tab20')  # tab20 gives 20 distinct colors
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(save_prefix + "_colored_overlay.png", bbox_inches='tight', dpi=150)
            plt.close()
            
            # Method 2: Contour overlay (each segment different colored contour)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(img_np)
            colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_classes)))
            for i, class_id in enumerate(unique_ids):
                if class_id == 0:  # skip background
                    continue
                mask_single = (mask_np == class_id).astype(np.uint8)
                color = colors[i % len(colors)][:3]  # RGB only
                ax.contour(mask_single, levels=[0.5], colors=[color], linewidths=2)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(save_prefix + "_colored_contours.png", bbox_inches='tight', dpi=150)
            plt.close()
            
            # Method 3: Side-by-side comparison with color legend
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original image
            axes[0].imshow(img_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Segmentation mask with colors
            im = axes[1].imshow(mask_np, cmap='tab20', vmin=0, vmax=20)
            axes[1].set_title('Segmentation')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(img_np)
            masked = np.ma.masked_where(mask_np == 0, mask_np)
            axes[2].imshow(masked, alpha=0.5, cmap='tab20', vmin=0, vmax=20)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            # Add colorbar for reference
            plt.colorbar(im, ax=axes[1], shrink=0.6, label='Segment ID')
            
            plt.tight_layout()
            plt.savefig(save_prefix + "_comparison.png", bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"[viz] Saved multi-class mask visualization: {save_prefix}_*.png")
            print(f"[viz] Found {num_classes} segments with IDs: {unique_ids[unique_ids > 0]}")
            print(f"[viz] Total segmented pixels: {np.sum(mask_np > 0)} out of {mask_np.size}")

        # Build a quick accessor to saved masks
        single_file_path = os.path.join(args.output_dir, "chexpert_masks.jsonl") if args.single_file else None

        print(f"[viz] Preparing visualization for {len(viz_candidates[: max(1, int(args.viz_num_samples))])} samples. Output dir: {args.viz_out_dir}")
        for i, (norm, split, abs_img) in enumerate(viz_candidates[: max(1, int(args.viz_num_samples))]):
            try:
                # Load saved mask based on chosen format
                mask_np = None
                if args.single_file and os.path.exists(single_file_path):
                    with open(single_file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            rec = json.loads(line)
                            if rec.get("path", "") == norm:
                                if "mask" in rec:
                                    # Multi-class mask stored as nested list
                                    mask_np = np.array(rec["mask"], dtype="int64")
                                elif "rle" in rec:
                                    # Binary mask stored as RLE (fallback)
                                    mask_np = rle_decode_binary(rec["rle"])
                                break
                else:
                    if args.jsonl:
                        mp = os.path.join(args.output_dir, os.path.splitext(norm)[0] + ".jsonl")
                        if os.path.exists(mp):
                            with open(mp, "r", encoding="utf-8") as f:
                                # take last line
                                last = None
                                for ln in f:
                                    last = ln
                                if last:
                                    rec = json.loads(last)
                                    mask_np = rle_decode_binary(rec["rle"])
                    else:
                        mp = os.path.join(args.output_dir, os.path.splitext(norm)[0] + ".pt")
                        if os.path.exists(mp):
                            m = torch.load(mp)
                            if isinstance(m, dict) and "mask" in m:
                                m = m["mask"]
                            m = torch.as_tensor(m).squeeze().cpu().numpy()
                            mask_np = (m > 0).astype("uint8")

                if mask_np is None:
                    print(f"[viz] Mask not found for path key: {norm}. Skipping.")
                    continue
                # Save overlay
                safe_name = norm.replace("/", "__").replace("\\", "__")
                save_prefix = os.path.join(args.viz_out_dir, f"viz_{i+1}_{safe_name}")
                print(f"[viz] Saving overlays to prefix: {save_prefix}")
                _overlay_and_save(abs_img, mask_np, save_prefix)
            except Exception as e:
                print(f"[viz] Failed to visualize {norm}: {e}")
                continue


if __name__ == "__main__":
    main()


