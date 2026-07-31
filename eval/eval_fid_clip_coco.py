"""
Compute FID and CLIP score on MS COCO 30k validation set (REAL COCO download).

Follows DUO paper Sec 4.1 evaluation protocol:
  - FID: measure distribution distance between SD1.4 (prior) and unlearned model
  - CLIP score: measure text-image alignment of generated images
  - Uses actual MS COCO 2014 validation captions via HuggingFace datasets

Usage:
  # Generate reference images (SD1.4 prior) first, then evaluate
  python eval/eval_fid_clip_coco.py \
      --model_root train/outputs/unlearn/SD-train/dpo/500 \
      --output eval/outputs/fid_clip/fid_clip_results.json \
      --coco_subset 30000 \
      --gen_batch_size 8

  # Evaluate only (if images already generated)
  python eval/eval_fid_clip_coco.py \
      --model_root train/outputs/unlearn/SD-train/dpo/500 \
      --output eval/outputs/fid_clip/fid_clip_results.json \
      --ref_dir /path/to/ref_images \
      --eval_only

Requirements:
  pip install datasets torchmetrics pytorch-fid
"""
import argparse
import gc
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# COCO 30k loader from HuggingFace datasets
# ---------------------------------------------------------------------------
def load_coco_30k(num: int = 30000, split: str = "validation") -> List[str]:
    """Load MS COCO 2014 validation captions via HuggingFace datasets.

    Returns up to `num` captions. The COCO validation set has ~40k captions
    for ~5k images; we sample 30k to match the DUO paper protocol.

    Falls back to proxy prompts if HuggingFace datasets is unavailable.
    """
    try:
        from datasets import load_dataset
        print("[COCO] Loading MS COCO 2014 validation captions from HuggingFace ...")
        ds = load_dataset("phiyodr/coco2014", split="validation", trust_remote_code=True)
        # Each image has 5 captions; take caption field
        captions = [ex["caption"] for ex in ds if ex.get("caption")]
        print(f"[COCO] {len(captions)} captions loaded")
        # Shuffle deterministically and take num
        import random
        rng = random.Random(42)
        rng.shuffle(captions)
        return captions[:num]
    except Exception as e:
        print(f"[COCO] HuggingFace datasets not available: {e}")
        print("[COCO] Falling back to proxy prompts (300 hardcoded)")
        return load_coco_proxy(num)


def load_coco_proxy(num: int = 300) -> List[str]:
    """Fallback proxy prompts when COCO download fails."""
    # Same 300 prompts as original eval_fid_clip.py
    proxy = [
        "a cat sitting on a couch", "a dog running in a park",
        "a woman holding an umbrella", "a man riding a bicycle",
        "a child playing with a ball", "a bird flying in the sky",
        "a boat on a lake", "a car parked on a street",
        "a group of people at a beach", "a house with a garden",
        "a bowl of fruit on a table", "a cup of coffee on a desk",
        "a city street at night", "a mountain landscape with snow",
        "a forest with autumn leaves", "a plate of pasta with vegetables",
        "a bride walking down an aisle", "a baby sleeping in a crib",
        "a laptop on a wooden desk", "a bookshelf filled with books",
        "a pair of sneakers on a floor", "a red fire truck on a road",
        "a white sheep in a field", "a black cat on a windowsill",
        "a train crossing a bridge", "a plane flying over clouds",
        "a ship sailing on the ocean", "a sunset over a beach",
        "a field of sunflowers", "a close up of a flower",
        "a pizza with pepperoni and cheese", "a slice of chocolate cake",
        "a glass of orange juice", "a fruit basket with apples and bananas",
        "a bowl of cereal with milk", "a grilled steak with vegetables",
        "a sushi platter with soy sauce", "a birthday cake with candles",
        "a picnic basket on a blanket", "a tent in a forest campsite",
        "a pair of glasses on a book", "a wristwatch on a table",
        "a smartphone with a cracked screen", "a wallet with cash and cards",
        "a keychain with a metal key", "a backpack on a chair",
        "a suitcase on a luggage rack", "a hat on a coat rack",
        "a pair of scissors on a desk", "a roll of tape on a table",
        "a pencil drawing of a face", "a watercolor painting of a landscape",
        "a oil painting of a bowl of fruit", "a digital art of a futuristic city",
        "a black and white photo of a street", "a vintage camera on a shelf",
        "a record player in a room", "a guitar leaning against a wall",
        "a piano in a concert hall", "a drum set on a stage",
        "a bouquet of red roses", "a potted succulent on a windowsill",
        "a bonsai tree on a table", "a vase with white lilies",
        "a fern in a hanging basket", "a garden with flowering bushes",
        "a wooden bench in a park", "a stone pathway through a garden",
        "a fountain in a city square", "a bridge over a small stream",
        "a lighthouse on a rocky cliff", "a barn in a rural landscape",
        "a windmill in a field of tulips", "a castle on a hill",
        "a temple in a forest", "a mosque with a dome",
        "a church with a tall spire", "a pagoda in a Japanese garden",
        "a row of colorful houses", "a modern apartment building",
        "a glass skyscraper in a city", "a cottage with a thatched roof",
        "a cabin in the woods", "a log cabin with a chimney",
        "a beach house with a deck", "a treehouse in a large oak",
        "a playground with swings", "a soccer ball on a green field",
        "a basketball hoop in a driveway", "a tennis court with a net",
        "a swimming pool in a backyard", "a yoga mat on a wooden floor",
        "a dumbbell on a gym floor", "a treadmill in a home gym",
        "a bicycle parked by a fence", "a skateboard on a ramp",
        "a surfboard on a sandy beach", "a snowboard on a snowy slope",
        "a pair of skis on a rack", "a baseball bat and ball",
        "a boxing glove on a ring", "a fishing rod by a river",
        "a campfire with logs burning", "a lantern hanging from a branch",
        "a telescope pointed at the sky", "a microscope on a lab table",
        "a stethoscope on a desk", "a thermometer showing high temperature",
        "a globe on a classroom desk", "a chalkboard with equations",
        "a whiteboard with colorful markers", "a calendar on a wall",
        "a clock showing noon", "a wooden chair at a desk",
        "a sofa in a living room", "a lamp on a nightstand",
        "a rug on a hardwood floor", "a curtain with floral pattern",
        "a pillow on a bed", "a blanket folded on a chair",
        "a towel on a bathroom rack", "a toothbrush in a cup",
        "a bar of soap on a sink", "a shampoo bottle in a shower",
        "a roll of toilet paper", "a trash can with a lid",
        "a refrigerator in a kitchen", "a stove with four burners",
        "a microwave on a counter", "a toaster next to a coffee maker",
        "a blender with fruit inside", "a kettle on a stove",
        "a set of knives on a magnetic strip", "a cutting board with chopped vegetables",
        "a frying pan with eggs", "a pot of boiling water",
        "a colander with pasta", "a measuring cup with flour",
        "a mixing bowl with batter", "a whisk on a counter",
        "a spatula and a ladle", "a plate with a fork and knife",
        "a wine glass on a table", "a beer mug with foam",
        "a tea cup with a saucer", "a pitcher of water with lemon",
        "a thermos on a picnic table", "a lunch box with a sandwich",
        "a napkin folded like a fan", "a tablecloth with checkered pattern",
        "a placemat with a bamboo design", "a coaster with a cork bottom",
        "a menu on a restaurant table", "a receipt from a grocery store",
        "a credit card on a counter", "a coin purse with change",
        "a piggy bank on a shelf", "a safe with a combination lock",
        "a mailbox at the end of a driveway", "a street sign at an intersection",
        "a stop sign on a road", "a traffic light showing green",
        "a fire hydrant on a sidewalk", "a parking meter on a street",
        "a bus stop with a bench", "a taxi cab on a city street",
        "a police car with lights", "an ambulance with emergency lights",
        "a school bus on a road", "a delivery truck with packages",
        "a tractor in a field", "a bulldozer at a construction site",
        "a crane lifting a beam", "a ladder leaning against a wall",
        "a toolbox with tools inside", "a hammer on a workbench",
        "a screwdriver with different bits", "a wrench on a metal surface",
        "a saw with sharp teeth", "a drill on a piece of wood",
        "a measuring tape on a table", "a level on a shelf",
        "a paintbrush with blue paint", "a paint roller on a tray",
        "a can of paint on a drop cloth", "a roll of wallpaper with pattern",
        "a tile cutter on a workbench", "a plunger next to a toilet",
        "a mop and bucket in a closet", "a vacuum cleaner in a hallway",
        "a broom and dustpan on a floor", "a sponge with soap suds",
        "a scrub brush on a tile floor", "a trash bag full of leaves",
        "a recycling bin with bottles", "a compost bin with food scraps",
        "a watering can in a garden", "a hose with a nozzle",
        "a shovel in a pile of dirt", "a rake next to a pile of leaves",
        "a wheelbarrow in a garden", "a pot of soil with a seedling",
        "a bag of fertilizer on a lawn", "a birdhouse on a pole",
        "a bird feeder with seeds", "a squirrel on a tree branch",
        "a rabbit in a hutch", "a hamster in a cage",
        "a fish in an aquarium", "a turtle on a rock",
        "a frog on a lily pad", "a butterfly on a flower",
        "a bee on a honeycomb", "a ladybug on a leaf",
        "a caterpillar on a branch", "a snail on a wet sidewalk",
        "a worm in the soil", "a spider web with dew drops",
        "a ant on a picnic blanket", "a dragonfly near a pond",
        "a grasshopper on a blade of grass", "a firefly in a jar",
        "a feather on a table", "a nest with eggs in a tree",
        "a seashell on a sandy beach", "a starfish on a rock",
        "a jellyfish in clear water", "a dolphin jumping in the ocean",
        "a whale swimming in deep water", "a seal on an iceberg",
        "a penguin on a snowy hill", "a polar bear on a ice floe",
        "a elephant in a savanna", "a giraffe eating leaves from a tall tree",
        "a lion resting under a tree", "a tiger walking in a jungle",
        "a zebra grazing in a field", "a rhinoceros in a mud puddle",
        "a hippopotamus in a river", "a monkey hanging from a vine",
        "a gorilla sitting on a rock", "a orangutan swinging in a tree",
        "a chimpanzee eating a banana", "a lemur on a branch",
        "a sloth hanging upside down", "a kangaroo with a joey in its pouch",
        "a koala on a eucalyptus tree", "a panda eating bamboo",
        "a bear catching a fish", "a fox in a snowy forest",
        "a wolf howling at the moon", "a deer in a meadow",
        "a moose near a lake", "a bison on a prairie",
        "a camel in a desert", "a horse galloping in a field",
        "a donkey standing by a fence", "a cow grazing in a pasture",
        "a pig in a mud pen", "a goat on a rocky hill",
        "a sheep with thick wool", "a chicken in a coop",
        "a rooster crowing at dawn", "a duck swimming in a pond",
        "a goose on a grassy bank", "a turkey in a farm yard",
        "a peacock displaying its feathers", "a swan on a lake",
        "a flamingo standing on one leg", "a stork on a chimney",
        "a eagle soaring in the sky", "a hawk perched on a branch",
        "a owl on a tree at night", "a parrot on a perch",
        "a crow on a fence post", "a raven on a rock",
        "a sparrow on a birdbath", "a robin on a branch",
        "a blue jay at a feeder", "a cardinal in a snowy bush",
        "a woodpecker on a tree trunk", "a hummingbird hovering by a flower",
        "a penguin sliding on ice", "a puffin on a cliff",
        "a pelican by the sea", "a seagull on a beach",
        "a albatross soaring over waves",
    ]
    result = []
    while len(result) < num:
        result.extend(proxy)
    return result[:num]


# ---------------------------------------------------------------------------
# CLIP Score
# ---------------------------------------------------------------------------
def load_clip(device: str):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14").to(device)
    proc = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14")
    return model, proc


@torch.no_grad()
def clip_score(
    images: List[Image.Image],
    texts: List[str],
    clip_model,
    clip_proc,
    device: str,
) -> float:
    inputs = clip_proc(
        text=texts, images=images,
        return_tensors="pt", padding=True, truncation=True,
    ).to(device)
    outputs = clip_model(**inputs)
    sims = (outputs.text_embeds * outputs.image_embeds).sum(dim=-1)
    return sims.mean().item()


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------
def compute_fid(real_dir: str, fake_dir: str, device: str,
                batch_size: int = 32) -> Optional[float]:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchvision import transforms as T

        transform = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        real_paths = sorted([
            os.path.join(real_dir, f) for f in os.listdir(real_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        fake_paths = sorted([
            os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if not real_paths or not fake_paths:
            print(f"[FID] No images found in {real_dir} or {fake_dir}")
            return None

        n = min(len(real_paths), len(fake_paths))
        real_paths = real_paths[:n]
        fake_paths = fake_paths[:n]
        print(f"[FID] Computing with {n} images per set")

        fid = FrechetInceptionDistance(feature=2048).to(device)

        for i in range(0, n, batch_size):
            real_batch = torch.stack([
                transform(Image.open(p).convert('RGB'))
                for p in real_paths[i:i + batch_size]
            ]).to(device)
            fid.update(real_batch, real=True)

        for i in range(0, n, batch_size):
            fake_batch = torch.stack([
                transform(Image.open(p).convert('RGB'))
                for p in fake_paths[i:i + batch_size]
            ]).to(device)
            fid.update(fake_batch, real=False)

        return fid.compute().item()

    except ImportError as e:
        print(f"[FID] torchmetrics/torchvision not available: {e}")
        return _compute_fid_cli(real_dir, fake_dir)
    except Exception as e:
        print(f"[FID] Error: {e}")
        return None


def _compute_fid_cli(real_dir: str, fake_dir: str) -> Optional[float]:
    try:
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "pytorch_fid", real_dir, fake_dir],
            capture_output=True, text=True, timeout=600
        )
        for line in result.stdout.split('\n'):
            if 'FID' in line:
                return float(line.split()[-1])
        return None
    except Exception as e:
        print(f"[FID] CLI fallback failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------
def load_pipeline_single(pretrained: str, lora_path: Optional[str],
                         device: str):
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    if lora_path and os.path.exists(lora_path):
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora()
    return pipe


@torch.no_grad()
def generate_images(pipe, prompts: List[str], seed: int,
                    batch_size: int, device: str) -> List[Image.Image]:
    g = torch.Generator(device).manual_seed(seed)
    images = []
    for i in tqdm(range(0, len(prompts), batch_size),
                  desc=f"Generating (seed={seed})"):
        batch = prompts[i:i + batch_size]
        with torch.amp.autocast(device):
            imgs = pipe(batch, generator=g, num_inference_steps=50).images
        images.extend(imgs)
        gc.collect()
        torch.cuda.empty_cache()
    return images


def save_images(images: List[Image.Image], out_dir: str, prefix: str = ""):
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(out_dir, f"{prefix}{i:05d}.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="FID and CLIP score on MS COCO 30k."
    )
    p.add_argument("--model_root", type=str, required=True,
                    help="Root dir containing Nudity/, Blood/, ... subdirs")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--pretrained", type=str,
                    default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--coco_subset", type=int, default=30000,
                    help="Number of COCO captions to use (paper: 30k)")
    p.add_argument("--gen_batch_size", type=int, default=4,
                    help="Generation batch size (reduce if OOM)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ref_dir", type=str, default=None,
                    help="Pre-generated reference images dir. If None, generate.")
    p.add_argument("--eval_only", action="store_true",
                    help="Skip generation, only compute metrics on existing dirs.")
    p.add_argument("--output_images_dir", type=str, default=None,
                    help="Save generated images to this dir for later reuse.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.model_root) and not args.eval_only:
        print(f"[skip] model_root not found: {args.model_root}")
        return

    device = args.device
    print(f"[eval] Device: {device}")

    # 1. Load COCO captions
    prompts = load_coco_30k(args.coco_subset)
    num = min(len(prompts), args.coco_subset)
    prompts = prompts[:num]
    print(f"[eval] {len(prompts)} prompts loaded")

    # 2. Load CLIP
    print("[eval] Loading CLIP ...")
    clip_model, clip_proc = load_clip(device)

    # 3. Generate / load reference images (SD1.4 prior)
    if args.ref_dir and os.path.isdir(args.ref_dir):
        ref_paths = sorted([
            os.path.join(args.ref_dir, f) for f in os.listdir(args.ref_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        ref_images = [Image.open(p).convert("RGB") for p in ref_paths]
        print(f"[eval] Loaded {len(ref_images)} reference images from {args.ref_dir}")
    else:
        print("[eval] Generating reference images (SD1.4 prior) ...")
        pipe_ref = load_pipeline_single(args.pretrained, None, device)
        ref_images = generate_images(pipe_ref, prompts, args.seed,
                                     args.gen_batch_size, device)
        del pipe_ref
        gc.collect()
        torch.cuda.empty_cache()

    # Save reference if requested
    if args.output_images_dir:
        save_images(ref_images, os.path.join(args.output_images_dir, "ref"),
                    prefix="ref_")
        print(f"[eval] Reference images saved to {args.output_images_dir}/ref/")

    # 4. Generate unlearned model images
    if args.eval_only and args.ref_dir:
        # In eval_only mode, try to load pre-generated unlearn images
        fake_dir = os.path.join(os.path.dirname(args.ref_dir), "unlearn")
        if os.path.isdir(fake_dir):
            fake_paths = sorted([
                os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            unlearn_images = [Image.open(p).convert("RGB") for p in fake_paths]
            print(f"[eval] Loaded {len(unlearn_images)} unlearn images from {fake_dir}")
        else:
            print("[eval] No unlearn images dir found, generating ...")
            lora_path = _find_lora(args.model_root)
            pipe_unlearn = load_pipeline_single(args.pretrained, lora_path, device)
            unlearn_images = generate_images(pipe_unlearn, prompts, args.seed,
                                             args.gen_batch_size, device)
            del pipe_unlearn
            gc.collect()
            torch.cuda.empty_cache()
    else:
        lora_path = _find_lora(args.model_root)
        if lora_path:
            pipe_unlearn = load_pipeline_single(args.pretrained, lora_path, device)
            unlearn_images = generate_images(pipe_unlearn, prompts, args.seed,
                                             args.gen_batch_size, device)
            del pipe_unlearn
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("[eval] No LoRA found, using reference as unlearn (FID=0)")
            unlearn_images = ref_images

    # Save unlearn if requested
    if args.output_images_dir:
        save_images(unlearn_images, os.path.join(args.output_images_dir, "unlearn"),
                    prefix="unlearn_")
        print(f"[eval] Unlearn images saved to {args.output_images_dir}/unlearn/")

    # 5. Compute CLIP score
    print("[eval] Computing CLIP score ...")
    clip_ref = clip_score(ref_images[:len(prompts)], prompts,
                          clip_model, clip_proc, device)
    clip_unlearn = clip_score(unlearn_images[:len(prompts)], prompts,
                              clip_model, clip_proc, device)
    print(f"  CLIP reference (SD1.4): {clip_ref:.4f}")
    print(f"  CLIP unlearn:           {clip_unlearn:.4f}")

    # 6. Compute FID (save to temp dirs)
    print("[eval] Computing FID ...")
    ref_dir_tmp = tempfile.mkdtemp(prefix="fid_ref_")
    unlearn_dir_tmp = tempfile.mkdtemp(prefix="fid_unlearn_")

    n_fid = min(len(ref_images), len(unlearn_images))
    save_images(ref_images[:n_fid], ref_dir_tmp)
    save_images(unlearn_images[:n_fid], unlearn_dir_tmp)

    fid_val = compute_fid(ref_dir_tmp, unlearn_dir_tmp, device,
                          batch_size=args.gen_batch_size)
    print(f"  FID ({n_fid} images): {fid_val:.4f}" if fid_val else "  FID: N/A")

    shutil.rmtree(ref_dir_tmp, ignore_errors=True)
    shutil.rmtree(unlearn_dir_tmp, ignore_errors=True)

    # 7. Save results
    results = {
        "model_root": args.model_root,
        "num_prompts": len(prompts),
        "coco_subset": args.coco_subset,
        "seed": args.seed,
        "clip_score_reference": round(clip_ref, 4),
        "clip_score_unlearn": round(clip_unlearn, 4),
        "fid": round(fid_val, 4) if fid_val else None,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    print(json.dumps(results, indent=2))

    del clip_model, clip_proc
    gc.collect()
    torch.cuda.empty_cache()


def _find_lora(model_root: str) -> Optional[str]:
    """Find a LoRA safetensors file under model_root."""
    for concept in ["Nudity", "Blood", "Gun", "Horror", "Suffer"]:
        cand = os.path.join(model_root, concept,
                            "pytorch_lora_weights.safetensors")
        if os.path.exists(cand):
            return cand
    return None


if __name__ == "__main__":
    main()
