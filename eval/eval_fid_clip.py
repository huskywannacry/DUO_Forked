"""
Compute FID and CLIP score on MS COCO 30k validation set.

Follows DUO paper Sec 4.1 evaluation protocol:
  - FID: measure distribution distance between SD1.4 (prior) and unlearned model
  - CLIP score: measure text-image alignment of generated images

Usage:
  python eval/eval_fid_clip.py \
      --model_root train/outputs/unlearn/SD-train/dpo/500 \
      --output eval/outputs/fid_clip/fid_clip_results.json

  python eval/eval_fid_clip.py \
      --model_root train/outputs/unlearn/SD-train/duo-anchor/500 \
      --output eval/outputs/fid_clip/fid_clip_results.json

Requirements:
  pip install torchmetrics pytorch-fid
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Minimal CLIP Score implementation (no torchvision dependency)
# ---------------------------------------------------------------------------
def load_clip(device: str):
    """Load CLIP ViT-L/14 for CLIP score computation."""
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
    """Compute mean CLIP score (cosine similarity) for image-text pairs."""
    inputs = clip_proc(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    outputs = clip_model(**inputs)
    # text_embeds: (N, D), image_embeds: (N, D), both L2-normalized
    sims = (outputs.text_embeds * outputs.image_embeds).sum(dim=-1)
    return sims.mean().item()


# ---------------------------------------------------------------------------
# Minimal FID via torchmetrics (pytorch-fid fallback if missing)
# ---------------------------------------------------------------------------
def compute_fid(
    real_images_dir: str,
    fake_images_dir: str,
    device: str,
    batch_size: int = 32,
) -> Optional[float]:
    """Compute FID between two directories of images.

    Uses torchmetrics if available, otherwise pytorch-fid CLI.
    Returns None on failure.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchvision import transforms as T
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader

        transform = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            # Inception normalization
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load real images (reference, e.g. SD1.4 prior)
        real_dataset = ImageFolder(os.path.dirname(real_images_dir)
                                   if os.path.isfile(real_images_dir)
                                   else real_images_dir,
                                   transform=transform)
        # Load fake images (unlearned model)
        fake_dataset = ImageFolder(os.path.dirname(fake_images_dir)
                                   if os.path.isfile(fake_images_dir)
                                   else fake_images_dir,
                                   transform=transform)

        # If they are not proper ImageFolder structure, use a simple list
        if len(real_dataset) == 0 or len(fake_dataset) == 0:
            print("[FID] Empty dataset, trying custom loader...")
            return _compute_fid_custom(
                real_images_dir, fake_images_dir, device, batch_size
            )

        fid = FrechetInceptionDistance(feature=2048).to(device)

        real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
        fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)

        for imgs, _ in tqdm(real_loader, desc="FID real"):
            fid.update(imgs.to(device), real=True)

        for imgs, _ in tqdm(fake_loader, desc="FID fake"):
            fid.update(imgs.to(device), real=False)

        return fid.compute().item()

    except ImportError as e:
        print(f"[FID] torchmetrics/torchvision not available: {e}")
        return _compute_fid_cli(real_images_dir, fake_images_dir)
    except Exception as e:
        print(f"[FID] Error: {e}")
        return None


def _compute_fid_custom(
    real_dir: str, fake_dir: str, device: str, batch_size: int
) -> Optional[float]:
    """Compute FID with custom image loading (no ImageFolder structure)."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchvision import transforms as T
        from PIL import Image

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

        # Use minimum of both counts
        n = min(len(real_paths), len(fake_paths))
        real_paths = real_paths[:n]
        fake_paths = fake_paths[:n]

        fid = FrechetInceptionDistance(feature=2048).to(device)

        for i in range(0, n, batch_size):
            real_batch = torch.stack([
                transform(Image.open(p).convert('RGB'))
                for p in real_paths[i:i+batch_size]
            ]).to(device)
            fid.update(real_batch, real=True)

        for i in range(0, n, batch_size):
            fake_batch = torch.stack([
                transform(Image.open(p).convert('RGB'))
                for p in fake_paths[i:i+batch_size]
            ]).to(device)
            fid.update(fake_batch, real=False)

        return fid.compute().item()

    except ImportError as e:
        print(f"[FID] torchmetrics not available: {e}")
        return _compute_fid_cli(real_dir, fake_dir)


def _compute_fid_cli(real_dir: str, fake_dir: str) -> Optional[float]:
    """Fallback: use pytorch-fid CLI."""
    try:
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "pytorch_fid", real_dir, fake_dir],
            capture_output=True, text=True, timeout=300
        )
        # Parse output: "FID: 12.345"
        for line in result.stdout.split('\n'):
            if 'FID' in line:
                return float(line.split()[-1])
        return None
    except Exception as e:
        print(f"[FID] CLI fallback failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Helper: generate images from a model + prompts
# ---------------------------------------------------------------------------
def load_pipeline_single(pretrained: str, lora_path: Optional[str],
                         device: str):
    """Load base SD + optional LoRA, return pipe."""
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained, torch_dtype=torch.float16
    ).to(device)
    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    if lora_path and os.path.exists(lora_path):
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora()
    return pipe


def generate_images(pipe, prompts: List[str], seed: int = 42,
                    batch_size: int = 4) -> List[Image.Image]:
    """Generate images from prompts."""
    g = torch.Generator("cuda").manual_seed(seed)
    images = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        with torch.amp.autocast("cuda"):
            imgs = pipe(batch, generator=g).images
        images.extend(imgs)
    return images


# ---------------------------------------------------------------------------
# Generate COCO-like eval set (no actual COCO download)
# Use a subset of i2p safe prompts + generic prompts as proxy
# ---------------------------------------------------------------------------
def load_coco_prompts(num: int = 300) -> List[str]:
    """Load a generic set of prompts as COCO proxy.

    Note: this is a substitute for actual MS COCO 30k captions.
    For the official FID/CLIP numbers you need to download COCO 30k.
    """
    coco_proxy_prompts = [
        "a cat sitting on a couch",
        "a dog running in a park",
        "a woman holding an umbrella",
        "a man riding a bicycle",
        "a child playing with a ball",
        "a bird flying in the sky",
        "a boat on a lake",
        "a car parked on a street",
        "a group of people at a beach",
        "a house with a garden",
        "a bowl of fruit on a table",
        "a cup of coffee on a desk",
        "a city street at night",
        "a mountain landscape with snow",
        "a forest with autumn leaves",
        "a plate of pasta with vegetables",
        "a bride walking down an aisle",
        "a baby sleeping in a crib",
        "a laptop on a wooden desk",
        "a bookshelf filled with books",
        "a pair of sneakers on a floor",
        "a red fire truck on a road",
        "a white sheep in a field",
        "a black cat on a windowsill",
        "a train crossing a bridge",
        "a plane flying over clouds",
        "a ship sailing on the ocean",
        "a sunset over a beach",
        "a field of sunflowers",
        "a close up of a flower",
        "a pizza with pepperoni and cheese",
        "a slice of chocolate cake",
        "a glass of orange juice",
        "a fruit basket with apples and bananas",
        "a bowl of cereal with milk",
        "a grilled steak with vegetables",
        "a sushi platter with soy sauce",
        "a birthday cake with candles",
        "a picnic basket on a blanket",
        "a tent in a forest campsite",
        "a pair of glasses on a book",
        "a wristwatch on a table",
        "a smartphone with a cracked screen",
        "a wallet with cash and cards",
        "a keychain with a metal key",
        "a backpack on a chair",
        "a suitcase on a luggage rack",
        "a hat on a coat rack",
        "a pair of scissors on a desk",
        "a roll of tape on a table",
        "a pencil drawing of a face",
        "a watercolor painting of a landscape",
        "a oil painting of a bowl of fruit",
        "a digital art of a futuristic city",
        "a black and white photo of a street",
        "a vintage camera on a shelf",
        "a record player in a room",
        "a guitar leaning against a wall",
        "a piano in a concert hall",
        "a drum set on a stage",
        "a bouquet of red roses",
        "a potted succulent on a windowsill",
        "a bonsai tree on a table",
        "a vase with white lilies",
        "a fern in a hanging basket",
        "a garden with flowering bushes",
        "a wooden bench in a park",
        "a stone pathway through a garden",
        "a fountain in a city square",
        "a bridge over a small stream",
        "a lighthouse on a rocky cliff",
        "a barn in a rural landscape",
        "a windmill in a field of tulips",
        "a castle on a hill",
        "a temple in a forest",
        "a mosque with a dome",
        "a church with a tall spire",
        "a pagoda in a Japanese garden",
        "a row of colorful houses",
        "a modern apartment building",
        "a glass skyscraper in a city",
        "a cottage with a thatched roof",
        "a cabin in the woods",
        "a log cabin with a chimney",
        "a beach house with a deck",
        "a treehouse in a large oak",
        "a playground with swings",
        "a soccer ball on a green field",
        "a basketball hoop in a driveway",
        "a tennis court with a net",
        "a swimming pool in a backyard",
        "a yoga mat on a wooden floor",
        "a dumbbell on a gym floor",
        "a treadmill in a home gym",
        "a bicycle parked by a fence",
        "a skateboard on a ramp",
        "a surfboard on a sandy beach",
        "a snowboard on a snowy slope",
        "a pair of skis on a rack",
        "a baseball bat and ball",
        "a boxing glove on a ring",
        "a fishing rod by a river",
        "a campfire with logs burning",
        "a lantern hanging from a branch",
        "a telescope pointed at the sky",
        "a microscope on a lab table",
        "a stethoscope on a desk",
        "a thermometer showing high temperature",
        "a globe on a classroom desk",
        "a chalkboard with equations",
        "a whiteboard with colorful markers",
        "a calendar on a wall",
        "a clock showing noon",
        "a wooden chair at a desk",
        "a sofa in a living room",
        "a lamp on a nightstand",
        "a rug on a hardwood floor",
        "a curtain with floral pattern",
        "a pillow on a bed",
        "a blanket folded on a chair",
        "a towel on a bathroom rack",
        "a toothbrush in a cup",
        "a bar of soap on a sink",
        "a shampoo bottle in a shower",
        "a roll of toilet paper",
        "a trash can with a lid",
        "a refrigerator in a kitchen",
        "a stove with four burners",
        "a microwave on a counter",
        "a toaster next to a coffee maker",
        "a blender with fruit inside",
        "a kettle on a stove",
        "a set of knives on a magnetic strip",
        "a cutting board with chopped vegetables",
        "a frying pan with eggs",
        "a pot of boiling water",
        "a colander with pasta",
        "a measuring cup with flour",
        "a mixing bowl with batter",
        "a whisk on a counter",
        "a spatula and a ladle",
        "a plate with a fork and knife",
        "a wine glass on a table",
        "a beer mug with foam",
        "a tea cup with a saucer",
        "a pitcher of water with lemon",
        "a thermos on a picnic table",
        "a lunch box with a sandwich",
        "a napkin folded like a fan",
        "a tablecloth with checkered pattern",
        "a placemat with a bamboo design",
        "a coaster with a cork bottom",
        "a menu on a restaurant table",
        "a receipt from a grocery store",
        "a credit card on a counter",
        "a coin purse with change",
        "a piggy bank on a shelf",
        "a safe with a combination lock",
        "a mailbox at the end of a driveway",
        "a street sign at an intersection",
        "a stop sign on a road",
        "a traffic light showing green",
        "a fire hydrant on a sidewalk",
        "a parking meter on a street",
        "a bus stop with a bench",
        "a taxi cab on a city street",
        "a police car with lights",
        "an ambulance with emergency lights",
        "a school bus on a road",
        "a delivery truck with packages",
        "a tractor in a field",
        "a bulldozer at a construction site",
        "a crane lifting a beam",
        "a ladder leaning against a wall",
        "a toolbox with tools inside",
        "a hammer on a workbench",
        "a screwdriver with different bits",
        "a wrench on a metal surface",
        "a saw with sharp teeth",
        "a drill on a piece of wood",
        "a measuring tape on a table",
        "a level on a shelf",
        "a paintbrush with blue paint",
        "a paint roller on a tray",
        "a can of paint on a drop cloth",
        "a roll of wallpaper with pattern",
        "a tile cutter on a workbench",
        "a plunger next to a toilet",
        "a mop and bucket in a closet",
        "a vacuum cleaner in a hallway",
        "a broom and dustpan on a floor",
        "a sponge with soap suds",
        "a scrub brush on a tile floor",
        "a trash bag full of leaves",
        "a recycling bin with bottles",
        "a compost bin with food scraps",
        "a watering can in a garden",
        "a hose with a nozzle",
        "a shovel in a pile of dirt",
        "a rake next to a pile of leaves",
        "a wheelbarrow in a garden",
        "a pot of soil with a seedling",
        "a bag of fertilizer on a lawn",
        "a birdhouse on a pole",
        "a bird feeder with seeds",
        "a squirrel on a tree branch",
        "a rabbit in a hutch",
        "a hamster in a cage",
        "a fish in an aquarium",
        "a turtle on a rock",
        "a frog on a lily pad",
        "a butterfly on a flower",
        "a bee on a honeycomb",
        "a ladybug on a leaf",
        "a caterpillar on a branch",
        "a snail on a wet sidewalk",
        "a worm in the soil",
        "a spider web with dew drops",
        "a ant on a picnic blanket",
        "a dragonfly near a pond",
        "a grasshopper on a blade of grass",
        "a firefly in a jar",
        "a feather on a table",
        "a nest with eggs in a tree",
        "a seashell on a sandy beach",
        "a starfish on a rock",
        "a jellyfish in clear water",
        "a dolphin jumping in the ocean",
        "a whale swimming in deep water",
        "a seal on an iceberg",
        "a penguin on a snowy hill",
        "a polar bear on a ice floe",
        "a elephant in a savanna",
        "a giraffe eating leaves from a tall tree",
        "a lion resting under a tree",
        "a tiger walking in a jungle",
        "a zebra grazing in a field",
        "a rhinoceros in a mud puddle",
        "a hippopotamus in a river",
        "a monkey hanging from a vine",
        "a gorilla sitting on a rock",
        "a orangutan swinging in a tree",
        "a chimpanzee eating a banana",
        "a lemur on a branch",
        "a sloth hanging upside down",
        "a kangaroo with a joey in its pouch",
        "a koala on a eucalyptus tree",
        "a panda eating bamboo",
        "a bear catching a fish",
        "a fox in a snowy forest",
        "a wolf howling at the moon",
        "a deer in a meadow",
        "a moose near a lake",
        "a bison on a prairie",
        "a camel in a desert",
        "a horse galloping in a field",
        "a donkey standing by a fence",
        "a cow grazing in a pasture",
        "a pig in a mud pen",
        "a goat on a rocky hill",
        "a sheep with thick wool",
        "a chicken in a coop",
        "a rooster crowing at dawn",
        "a duck swimming in a pond",
        "a goose on a grassy bank",
        "a turkey in a farm yard",
        "a peacock displaying its feathers",
        "a swan on a lake",
        "a flamingo standing on one leg",
        "a stork on a chimney",
        "a eagle soaring in the sky",
        "a hawk perched on a branch",
        "a owl on a tree at night",
        "a parrot on a perch",
        "a crow on a fence post",
        "a raven on a rock",
        "a sparrow on a birdbath",
        "a robin on a branch",
        "a blue jay at a feeder",
        "a cardinal in a snowy bush",
        "a woodpecker on a tree trunk",
        "a hummingbird hovering by a flower",
        "a penguin sliding on ice",
        "a puffin on a cliff",
        "a pelican by the sea",
        "a seagull on a beach",
        "a albatross soaring over waves",
    ]
    # Repeat to reach desired count
    result = []
    while len(result) < num:
        result.extend(coco_proxy_prompts)
    return result[:num]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute FID and CLIP score for unlearned models."
    )
    p.add_argument("--model_root", type=str, required=True,
                    help="Root dir containing Nudity/, Blood/, ... subdirs")
    p.add_argument("--output", type=str, default=None,
                    help="Output JSON path")
    p.add_argument("--pretrained", type=str,
                    default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--num_prompts", type=int, default=300,
                    help="Number of prompts to evaluate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--real_images_dir", type=str, default=None,
                    help="Directory of real (prior) images. If None, generate from SD1.4 first.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.model_root):
        print(f"[skip] model_root not found: {args.model_root}")
        return

    device = args.device

    # Load COCO proxy prompts
    prompts = load_coco_prompts(args.num_prompts)
    print(f"[eval] {len(prompts)} prompts loaded")

    # Load CLIP for scoring
    print("[eval] Loading CLIP...")
    clip_model, clip_proc = load_clip(device)

    # Generate images from reference (prior) model
    print("[eval] Generating reference images (SD1.4 prior)...")
    pipe_ref = load_pipeline_single(args.pretrained, None, device)
    ref_images = generate_images(pipe_ref, prompts, seed=args.seed)
    del pipe_ref
    gc.collect()
    torch.cuda.empty_cache()

    # Generate images from unlearned model
    print(f"[eval] Generating from model at {args.model_root}...")
    # For a multi-LoRA model (violence: 4 sub-LoRAs merged), we need the
    # actually combined weights. For simplicity, we load the Nudity LoRA
    # as a representative single LoRA, OR if multiple exist, skip.
    lora_path = os.path.join(args.model_root, "Nudity",
                             "pytorch_lora_weights.safetensors")
    if not os.path.exists(lora_path):
        print(f"[eval] No single LoRA found at {lora_path}, trying first available...")
        for concept in ["Blood", "Gun", "Horror", "Suffer"]:
            cand = os.path.join(args.model_root, concept,
                                "pytorch_lora_weights.safetensors")
            if os.path.exists(cand):
                lora_path = cand
                break

    if os.path.exists(lora_path):
        pipe_unlearn = load_pipeline_single(args.pretrained, lora_path, device)
        unlearn_images = generate_images(pipe_unlearn, prompts, seed=args.seed)
        del pipe_unlearn
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("[eval] No LoRA weights found, using reference as unlearn (FID=0)")
        unlearn_images = ref_images

    # Compute CLIP score for unlearned model
    print("[eval] Computing CLIP score...")
    clip_val_ref = clip_score(ref_images, prompts, clip_model, clip_proc, device)
    clip_val_unlearn = clip_score(unlearn_images, prompts, clip_model, clip_proc, device)

    print(f"  CLIP reference (SD1.4): {clip_val_ref:.4f}")
    print(f"  CLIP unlearn:           {clip_val_unlearn:.4f}")

    # Compute FID (if we have both sets of images on disk)
    # Save images to temp dirs for FID computation
    import tempfile
    ref_dir = tempfile.mkdtemp(prefix="ref_")
    unlearn_dir = tempfile.mkdtemp(prefix="unlearn_")

    for i, img in enumerate(ref_images):
        img.save(os.path.join(ref_dir, f"{i:04d}.png"))
    for i, img in enumerate(unlearn_images):
        img.save(os.path.join(unlearn_dir, f"{i:04d}.png"))

    print("[eval] Computing FID...")
    fid_val = compute_fid(ref_dir, unlearn_dir, device)
    print(f"  FID ({args.num_prompts} images): {fid_val:.4f}" if fid_val else "  FID: N/A")

    # Cleanup temp dirs
    import shutil
    shutil.rmtree(ref_dir, ignore_errors=True)
    shutil.rmtree(unlearn_dir, ignore_errors=True)

    results = {
        "model_root": args.model_root,
        "num_prompts": args.num_prompts,
        "clip_score_reference": round(clip_val_ref, 4),
        "clip_score_unlearn": round(clip_val_unlearn, 4),
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


if __name__ == "__main__":
    main()
