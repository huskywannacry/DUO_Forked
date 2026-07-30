"""
Enhanced anchor image generator.

Extends generate_anchors.py with:
  1. CLIP image-similarity filtering — pick images closest to the harmful concept
     in image embedding space (hard anchors).
  2. NudeNet safety filter — reject images that themselves contain nudity/violence.
  3. Diversity selection — ensure visual diversity across the selected anchor set.
  4. Writes anchor embeddings metadata for analysis.

Usage:
  python datasets/SD/generate_anchors_enhanced.py \
      --config datasets/SD/config_auto_anchor.json \
      --data_dir datasets/SD \
      --per_prompt 6 \
      --device cuda:0

Or use the wrapper script:
  bash scripts/prepare-anchor-enhanced.sh
"""
import argparse
import gc
import json
import os
import sys
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_clip(device: str):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, proc


@torch.no_grad()
def embed_images(model, proc, images: List[Image.Image], device: str) -> torch.Tensor:
    """Return normalized image embeddings (N, D)."""
    inputs = proc(images=images, return_tensors="pt", padding=True).to(device)
    feats = model.get_image_features(**inputs)
    return F.normalize(feats, p=2, dim=-1)


@torch.no_grad()
def embed_texts(model, proc, texts: List[str], device: str) -> torch.Tensor:
    """Return normalized text embeddings (N, D)."""
    inputs = proc(text=texts, return_tensors="pt", padding=True,
                  truncation=True).to(device)
    feats = model.get_text_features(**inputs)
    return F.normalize(feats, p=2, dim=-1)


def select_hard_anchors(
    images: List[Image.Image],
    prompts: List[str],
    image_embs: torch.Tensor,
    target_emb: torch.Tensor,
    clip_model, clip_proc,
    device: str,
    max_select: int = 32,
    sim_min: float = 0.4,
    sim_max: float = 0.90,
    diversity_threshold: float = 0.97,
) -> Tuple[List[Image.Image], List[str], List[float]]:
    """Select hard anchors based on CLIP image similarity + diversity.

    Returns (selected_images, selected_prompts, selected_similarities).
    """
    sims = (image_embs @ target_emb.T).squeeze(-1).tolist()

    # Filter by similarity range
    candidates = [
        (img, prompt, s) for img, prompt, s in zip(images, prompts, sims)
        if sim_min <= s <= sim_max
    ]
    candidates.sort(key=lambda x: x[2], reverse=True)

    if not candidates:
        # Fallback: take everything within 0.3-0.95
        candidates = [
            (img, prompt, s) for img, prompt, s in zip(images, prompts, sims)
            if 0.3 <= s <= 0.95
        ]
        candidates.sort(key=lambda x: x[2], reverse=True)

    if not candidates:
        return [], [], []

    # Diversity selection: greedily pick images with low mutual similarity
    selected_images, selected_prompts, selected_sims = [], [], []
    selected_embs = []

    for img, prompt, s in candidates:
        if len(selected_images) >= max_select:
            break

        # Enforce image-level diversity
        if selected_embs:
            img_emb = embed_images(clip_model, clip_proc, [img], device)
            mutual_sims = (img_emb @ torch.stack(selected_embs).T).max().item()
            if mutual_sims > diversity_threshold:
                continue  # Too similar to existing selection

        selected_images.append(img)
        selected_prompts.append(prompt)
        selected_sims.append(s)

        img_emb = embed_images(clip_model, clip_proc, [img], device)
        selected_embs.append(img_emb.squeeze(0))

    return selected_images, selected_prompts, selected_sims


def nudity_filter(images: List[Image.Image], device: str) -> List[bool]:
    """NudeNet filter. Returns [True if SAFE, False if NSFW]."""
    try:
        import nudenet
        detector = nudenet.NudeDetector(device=device if 'cuda' in device else 'cpu')
        safe_flags = []
        for img in tqdm(images, desc="NudeNet filter"):
            safe = True
            dets = detector.detect(img)
            for d in dets:
                label = d.get('class', '')
                if any(kw in label.upper() for kw in [
                    'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED',
                    'ANUS_EXPOSED', 'MALE_GENITALIA_EXPOSED',
                ]):
                    safe = False
                    break
            safe_flags.append(safe)
        del detector
        gc.collect()
        torch.cuda.empty_cache()
        return safe_flags
    except Exception as e:
        print(f"[WARN] NudeNet filter failed: {e}. Skipping.")
        return [True] * len(images)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default="datasets/SD/config_auto_anchor.json")
    p.add_argument("--data_dir", type=str, default="datasets/SD")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--per_prompt", type=int, default=6,
                   help="Number of images per anchor prompt to generate.")
    p.add_argument("--max_anchors", type=int, default=24,
                   help="Maximum anchor images per concept (after filtering).")
    p.add_argument("--pretrained_model_name_or_path",
                   type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--sim_min", type=float, default=0.4,
                   help="Min CLIP image similarity to target (exclude noisy images).")
    p.add_argument("--sim_max", type=float, default=0.90,
                   help="Max CLIP image similarity to target (exclude near-duplicate).")
    p.add_argument("--diversity_threshold", type=float, default=0.97,
                   help="Max mutual CLIP similarity (lower = more diverse).")
    p.add_argument("--nudenet_filter", action="store_true",
                   help="Use NudeNet to filter out NSFW-generated images.")
    p.add_argument("--skip_generation", action="store_true",
                   help="Re-filter existing images without generating new ones.")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = args.device
    weight_dtype = torch.float16 if "cuda" in device else torch.float32

    # Load CLIP once for all concepts
    print(f"[enhanced] Loading CLIP on {device} ...")
    clip_model, clip_proc = load_clip(device)

    if not args.skip_generation:
        # Load SD pipe once
        print(f"[enhanced] Loading SD {args.pretrained_model_name_or_path} ...")
        pipe = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, torch_dtype=weight_dtype
        ).to(device)
        pipe.safety_checker = None
        pipe.set_progress_bar_config(disable=True)

    for concept_name, ccfg in cfg.items():
        if "anchor_prompts" not in ccfg:
            continue

        anchor_prompts = ccfg["anchor_prompts"]
        anchor_dir_name = ccfg.get("anchor_images", "anchor")
        out_dir = os.path.join(args.data_dir, concept_name, anchor_dir_name)
        os.makedirs(out_dir, exist_ok=True)

        # Check existing images
        existing_paths = sorted([
            os.path.join(out_dir, f) for f in os.listdir(out_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]) if os.path.isdir(out_dir) else []

        total_anchor_needed = len(anchor_prompts) * args.per_prompt

        if args.skip_generation:
            print(f"\n[{concept_name}] re-filtering {len(existing_paths)} existing images ...")
            raw_images = [Image.open(p).convert("RGB") for p in existing_paths]
            # Map images back to prompts based on naming convention
            raw_prompts = anchor_prompts * args.per_prompt
            raw_prompts = raw_prompts[:len(raw_images)]
        else:
            if len(existing_paths) >= total_anchor_needed:
                print(f"\n[{concept_name}] already has {len(existing_paths)} images, "
                      f"skipping generation (use --skip_generation to re-filter).")
                continue

            print(f"\n[{concept_name}] generating {total_anchor_needed} images "
                  f"({len(anchor_prompts)} prompts × {args.per_prompt}) ...")

            raw_images = []
            raw_prompts = []
            idx = 0
            for prompt in tqdm(anchor_prompts, desc=f"{concept_name}"):
                for _ in range(args.per_prompt):
                    generator = torch.Generator(device).manual_seed(42 + idx)
                    img = pipe(prompt, num_images_per_prompt=1,
                               generator=generator).images[0]
                    # Save raw to disk immediately for resume
                    raw_img_path = os.path.join(out_dir, f"raw_{idx:03d}.jpg")
                    img.save(raw_img_path)

                    raw_images.append(img)
                    raw_prompts.append(prompt)
                    idx += 1

            print(f"[{concept_name}] generated {len(raw_images)} raw images")

        if not raw_images:
            print(f"[{concept_name}] no images to process, skipping.")
            continue

        # ---- Step 1: NudeNet safety filter ----
        if args.nudenet_filter:
            print(f"[{concept_name}] running NudeNet safety filter ...")
            safe_flags = nudity_filter(raw_images, device)
            safe_count = sum(safe_flags)
            print(f"  {safe_count}/{len(safe_flags)} images passed safety filter")
            raw_images = [img for img, f in zip(raw_images, safe_flags) if f]
            raw_prompts = [p for p, f in zip(raw_prompts, safe_flags) if f]

        if not raw_images:
            print(f"[{concept_name}] all images filtered out, skipping.")
            continue

        # ---- Step 2: CLIP hard anchor selection ----
        # Target embedding = mean of unsafe prompt embeddings
        target_prompts = ccfg.get("prompt", [])
        if isinstance(target_prompts, str):
            target_prompts = [target_prompts]

        target_emb = embed_texts(clip_model, clip_proc, target_prompts, device).mean(
            dim=0, keepdim=True)

        print(f"[{concept_name}] selecting hard anchors from {len(raw_images)} images ...")
        image_embs = embed_images(clip_model, clip_proc, raw_images, device)

        selected_imgs, selected_prompts, selected_sims = select_hard_anchors(
            raw_images, raw_prompts, image_embs, target_emb,
            clip_model, clip_proc, device,
            max_select=args.max_anchors,
            sim_min=args.sim_min,
            sim_max=args.sim_max,
            diversity_threshold=args.diversity_threshold,
        )

        print(f"  Selected {len(selected_imgs)} hard anchors "
              f"(sim range: {min(selected_sims):.3f} ~ {max(selected_sims):.3f})")

        # ---- Step 3: Save final anchors ----
        # Clear old files, keep only selected ones
        for fname in os.listdir(out_dir):
            fpath = os.path.join(out_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)

        # Save with prompt metadata
        anchor_metadata = []
        for i, (img, prompt, sim) in enumerate(zip(
                selected_imgs, selected_prompts, selected_sims)):
            save_path = os.path.join(out_dir, f"{i:03d}.jpg")
            img.save(save_path)
            anchor_metadata.append({
                "file": f"{i:03d}.jpg",
                "prompt": prompt,
                "clip_sim_to_target": round(sim, 4),
            })

        # Write metadata
        meta_path = os.path.join(out_dir, "_anchor_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(anchor_metadata, f, indent=2)

        print(f"[{concept_name}] saved {len(selected_imgs)} final anchors to {out_dir}")

        # Update config with actual counts
        ccfg["anchor_images"] = anchor_dir_name
        ccfg["_anchor_count"] = len(selected_imgs)
        ccfg["_anchor_sim_range"] = [
            round(min(selected_sims), 3),
            round(max(selected_sims), 3),
        ] if selected_sims else [0, 0]

    # Save updated config
    with open(args.config, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n[enhanced] Updated config written to {args.config}")

    # Cleanup
    del clip_model, clip_proc
    gc.collect()
    torch.cuda.empty_cache()
    if not args.skip_generation:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
