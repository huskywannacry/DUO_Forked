# Generate anchor images for Idea 1 (Anchor-based Retention)
# Anchor set = images of safe concepts visually similar to the harmful concept
# (e.g. for "Blood" unlearning: ketchup, tomato sauce, red wine, ...)
# Generated using the original (un-unlearned) SD 1.4 so the unlearned LoRA
# is forced to preserve them.
import argparse
import json
import os
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--config_dir", type=str, default="datasets/SD/config.json")
parser.add_argument("--data_dir", type=str, default="datasets/SD")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--per_prompt", type=int, default=4,
                    help="Number of images generated per anchor prompt")
parser.add_argument("--pretrained_model_name_or_path",
                    type=str, default="CompVis/stable-diffusion-v1-4")
args = parser.parse_args()

device = args.device
weight_dtype = torch.float16

with open(args.config_dir, "r") as f:
    cfg = json.load(f)

pipe = StableDiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path, torch_dtype=weight_dtype
).to(device)
pipe.safety_checker = None
pipe.set_progress_bar_config(disable=True)

for concept, ccfg in cfg.items():
    if "anchor_prompts" not in ccfg or "anchor_images" not in ccfg:
        continue
    out_dir = os.path.join(args.data_dir, concept, ccfg["anchor_images"])
    os.makedirs(out_dir, exist_ok=True)

    prompts = ccfg["anchor_prompts"]
    print(f"[{concept}] generating {len(prompts) * args.per_prompt} anchor images ...")
    idx = 0
    for prompt in tqdm(prompts, desc=f"{concept}"):
        generator = torch.Generator(device).manual_seed(42 + idx)
        for _ in range(args.per_prompt):
            save_path = os.path.join(out_dir, f"{idx:03d}.jpg")
            if os.path.exists(save_path):
                idx += 1
                continue
            img = pipe(prompt, num_images_per_prompt=1, generator=generator).images[0]
            img.save(save_path)
            idx += 1

print("Done.")
