"""
Auto-discover anchor prompts for each harmful concept using CLIP text space.

Given a harmful concept (e.g. "Blood"), this script:
  1. Queries a large noun phrase dictionary (built from common captions)
  2. Ranks candidates by cosine similarity to the target concept in CLIP text space
  3. Selects top-N candidates in the "hard anchor zone" (similarity ~0.6-0.85)
  4. Filters out candidates that are themselves unsafe (via NudeNet keyword blocklist)
  5. Writes discovered anchors back to config.json (or a YAML sidecar)

Usage:
  python datasets/SD/discover_anchors.py \
      --config datasets/SD/config.json \
      --output datasets/SD/config_auto_anchor.json \
      --top_k 20 \
      --similarity_range 0.60 0.85 \
      --device cuda:0
"""
import argparse
import json
import os
import sys
from typing import List, Tuple, Dict

import torch


# ---------------------------------------------------------------------------
# Default safe-category dictionary (in case the external file is missing)
# ---------------------------------------------------------------------------
BUILTIN_WORD_POOL = [
    # ---- Red / liquid / organic (Blood anchors) ----
    "a bottle of ketchup on a table, realistic photo",
    "a bowl of tomato sauce on a counter, realistic photo",
    "a jar of strawberry jam, realistic photo",
    "a splash of red wine on a white cloth, realistic photo",
    "a ripe pomegranate halved, realistic photo",
    "a plate of spaghetti bolognese, realistic photo",
    "a glass of red fruit punch, realistic photo",
    "a red brick wall, realistic photo",
    "a can of tomato soup, realistic photo",
    "a bowl of cherries, realistic photo",
    "a glass of cranberry juice, realistic photo",
    "a red velvet cake on a plate, realistic photo",
    "a bunch of red roses in a vase, realistic photo",
    "a red pepper sliced open, realistic photo",
    "a strawberry on a white plate, realistic photo",
    "a watermelon slice on a table, realistic photo",
    "a bowl of spicy chili con carne, realistic photo",
    "a red sports car on the road, realistic photo",
    "a red apple on a wooden table, realistic photo",
    "a fresh tomato on a cutting board, realistic photo",
    "a bottle of hot sauce on a table, realistic photo",
    "a bowl of raspberry sorbet, realistic photo",
    "a red coral reef underwater, realistic photo",
    "a red maple leaf in autumn, realistic photo",
    "a red dress on a mannequin, realistic photo",
    "a red traffic light on a street, realistic photo",
    "a cup of hibiscus tea, realistic photo",
    "a lobster on a plate, realistic photo",
    "a pile of red chili peppers, realistic photo",
    "a red rose tattoo on skin, realistic photo",
    "a red fire hydrant on a sidewalk, realistic photo",
    "a red sunset over the ocean, realistic photo",
    "a bowl of beetroot salad, realistic photo",
    "a red balloon floating in the sky, realistic photo",
    "a glass of red wine on a dinner table, realistic photo",
    "a strawberry milkshake in a glass, realistic photo",
    "a red cardinal bird on a branch, realistic photo",
    "a tomato plant in a garden, realistic photo",
    "a red poppy flower in a field, realistic photo",
    "a bowl of gochujang sauce, realistic photo",
    "a red enamel pot on a stove, realistic photo",
    "a red backpack on a chair, realistic photo",
    "a can of coke on a table, realistic photo",
    "a red guitar on a stand, realistic photo",
    "a bowl of tomato basil soup, realistic photo",
    "a red and white checkered tablecloth, realistic photo",
    "a bottle of tabasco sauce, realistic photo",
    "a red leather wallet on a desk, realistic photo",
    "a red solo cup on a table, realistic photo",
    "a red truck on a dirt road, realistic photo",
    "a red cabbage sliced in half, realistic photo",
    "a bowl of red curry, realistic photo",
    "a red lantern at a festival, realistic photo",

    # ---- Violence-adjacent but safe (Gun / weapon-shaped objects) ----
    "a person holding a metal wrench, realistic photo",
    "a carpenter with a toolbox, realistic photo",
    "a person carrying a backpack, realistic photo",
    "a photographer with a camera with a long lens, realistic photo",
    "a musician playing a violin, realistic photo",
    "a chef holding a kitchen knife while cutting vegetables, realistic photo",
    "a tennis player holding a racket, realistic photo",
    "a singer with a microphone on stage, realistic photo",
    "a person holding a flashlight, realistic photo",
    "a construction worker with a drill, realistic photo",
    "a person holding a garden hose, realistic photo",
    "a person with a walking stick on a hike, realistic photo",
    "a painter holding a long brush, realistic photo",
    "a person holding an umbrella, realistic photo",
    "a person holding a fishing rod, realistic photo",
    "a person holding a rolling pin in a kitchen, realistic photo",
    "a person holding a hair dryer, realistic photo",
    "a person holding a broom, realistic photo",
    "a person holding a cell phone, realistic photo",
    "a person holding a TV remote, realistic photo",
    "a person holding a spatula, realistic photo",
    "a person holding a screwdriver, realistic photo",
    "a person holding a hammer, realistic photo",
    "a person holding a baseball bat, realistic photo",
    "a person holding a hockey stick, realistic photo",
    "a person holding a pool cue, realistic photo",
    "a person holding a conductor baton, realistic photo",
    "a person holding a chopstick, realistic photo",
    "a person holding a pen, realistic photo",
    "a person holding a book, realistic photo",
    "a person holding a tablet, realistic photo",
    "a person with a bandaged arm, realistic photo",
    "a person wearing a neck brace, realistic photo",
    "a person using crutches, realistic photo",
    "a person in a wheelchair, realistic photo",
    "a person wearing a face mask at a hospital, realistic photo",
    "a person getting a vaccination injection, realistic photo",
    "a person with an ice pack on their knee, realistic photo",
    "a stethoscope on a doctor desk, realistic photo",
    "a person having a dental checkup, realistic photo",

    # ---- Horror-adjacent but safe (dark / spooky settings) ----
    "a dark forest at night, cinematic photo",
    "a creepy abandoned house on a hill, cinematic photo",
    "a foggy graveyard with old tombstones, cinematic photo",
    "a shadowy figure in a dark hallway, cinematic photo",
    "a close-up of a skull on a wooden table, cinematic photo",
    "a haunted mansion under a full moon, cinematic photo",
    "a mysterious black cat on a windowsill, cinematic photo",
    "a stormy night with lightning over a castle, cinematic photo",
    "a dark alleyway at night, cinematic photo",
    "a spiderweb with dew drops in the morning, cinematic photo",
    "a cave entrance in a mountainside, cinematic photo",
    "a broken window in an old building, cinematic photo",
    "a cemetery with fog rolling in, cinematic photo",
    "a dimly lit library with tall bookshelves, cinematic photo",
    "a forest path covered in mist, cinematic photo",
    "a deserted amusement park at night, cinematic photo",
    "a single candle in a dark room, cinematic photo",
    "a silhouette of a tree against the moon, cinematic photo",
    "an old castle ruin on a cliff, cinematic photo",
    "a cobweb-covered chandelier, cinematic photo",
    "a rusty iron gate in a stone wall, cinematic photo",
    "a dark basement with old furniture, cinematic photo",
    "a bat hanging from a cave ceiling, cinematic photo",
    "an owl on a branch at night, cinematic photo",
    "a howling wolf silhouetted by the moon, cinematic photo",
    "a raven perched on a fence, cinematic photo",
    "a field of dead trees under a grey sky, cinematic photo",
    "a foggy lighthouse on a rocky shore, cinematic photo",
    "a staircase leading into darkness, cinematic photo",
    "a broken mirror in an abandoned room, cinematic photo",

    # ---- Suffer-adjacent but safe (pain / sadness without actual suffering) ----
    "a crying baby with tears, realistic photo",
    "a person with a headache holding their head, realistic photo",
    "an exhausted runner after a marathon, realistic photo",
    "a sad child sitting alone on a bench, realistic photo",
    "a person sneezing into a tissue, realistic photo",
    "a person with a fever lying in bed, realistic photo",
    "a tired office worker sleeping at a desk, realistic photo",
    "a person with a stomach ache, realistic photo",
    "a child with a scraped knee getting a bandage, realistic photo",
    "a person coughing into their elbow, realistic photo",
    "a person with a cold drinking tea, realistic photo",
    "a person covered in mud after a fall, realistic photo",
    "a person shivering in cold weather, realistic photo",
    "a person crying tears of joy at a wedding, realistic photo",
    "a person sweating after a workout, realistic photo",
    "a person with a sunburn on the beach, realistic photo",
    "a person yawning widely, realistic photo",
    "a person with a bandaged finger, realistic photo",
    "a person wearing glasses reading a book, realistic photo",
    "a person blowing out candles on a birthday cake, realistic photo",

    # ---- General / neutral (catch-all to improve diversity) ----
    "a bowl of green salad on a table, realistic photo",
    "a cup of coffee with steam, realistic photo",
    "a dog sleeping on a carpet, realistic photo",
    "a cat sitting on a windowsill, realistic photo",
    "a person walking their dog in a park, realistic photo",
    "a bicycle parked against a fence, realistic photo",
    "a bookshelf filled with books, realistic photo",
    "a plate of pancakes with syrup, realistic photo",
    "a vase of fresh flowers on a table, realistic photo",
    "a family having a picnic in a park, realistic photo",
    "a child flying a kite on a beach, realistic photo",
    "a couple holding hands on a bridge, realistic photo",
    "a chef cooking in a kitchen, realistic photo",
    "a musician playing a guitar, realistic photo",
    "a painter at an easel in a studio, realistic photo",
    "a person reading a newspaper, realistic photo",
    "a group of friends laughing at a cafe, realistic photo",
    "a dancer performing on stage, realistic photo",
    "a person gardening in a backyard, realistic photo",
    "a fisherman on a boat at sunset, realistic photo",
]


# ---------------------------------------------------------------------------
# Hard-coded synonyms and "close but safe" prompts for each concept
# (used when --use_builtin_only is set)
# ---------------------------------------------------------------------------
BUILTIN_ANCHOR_MAP = {
    "Blood": [
        "a bottle of ketchup on a table, realistic photo",
        "a bowl of tomato sauce on a counter, realistic photo",
        "a jar of strawberry jam, realistic photo",
        "a splash of red wine on a white cloth, realistic photo",
        "a ripe pomegranate halved, realistic photo",
        "a plate of spaghetti bolognese, realistic photo",
        "a glass of red fruit punch, realistic photo",
        "a red brick wall, realistic photo",
        "a can of tomato soup, realistic photo",
        "a bowl of cherries, realistic photo",
        "a glass of cranberry juice, realistic photo",
        "a red velvet cake on a plate, realistic photo",
        "a bunch of red roses in a vase, realistic photo",
        "a red pepper sliced open, realistic photo",
        "a watermelon slice, realistic photo",
        "a fresh tomato on a cutting board, realistic photo",
        "a bottle of hot sauce on a table, realistic photo",
        "a bowl of raspberry sorbet, realistic photo",
        "a bowl of beetroot salad, realistic photo",
        "a red maple leaf in autumn, realistic photo",
    ],
    "Gun": [
        "a person holding a metal wrench, realistic photo",
        "a carpenter with a toolbox, realistic photo",
        "a person carrying a backpack, realistic photo",
        "a photographer with a camera with a long lens, realistic photo",
        "a chef holding a kitchen knife while cutting vegetables, realistic photo",
        "a tennis player holding a racket, realistic photo",
        "a person holding a flashlight, realistic photo",
        "a construction worker with a drill, realistic photo",
        "a person holding a garden hose, realistic photo",
        "a person with a walking stick on a hike, realistic photo",
        "a painter holding a long brush, realistic photo",
        "a person holding an umbrella, realistic photo",
        "a person holding a fishing rod, realistic photo",
        "a person holding a rolling pin in a kitchen, realistic photo",
        "a person holding a hammer, realistic photo",
        "a person holding a baseball bat, realistic photo",
        "a person holding a hockey stick, realistic photo",
        "a person holding a pool cue, realistic photo",
        "a person holding a conductor baton, realistic photo",
        "a person holding a screwdriver, realistic photo",
    ],
    "Horror": [
        "a dark forest at night, cinematic photo",
        "a creepy abandoned house on a hill, cinematic photo",
        "a foggy graveyard with old tombstones, cinematic photo",
        "a shadowy figure in a dark hallway, cinematic photo",
        "a close-up of a skull on a wooden table, cinematic photo",
        "a haunted mansion under a full moon, cinematic photo",
        "a stormy night with lightning over a castle, cinematic photo",
        "a dark alleyway at night, cinematic photo",
        "a cave entrance in a mountainside, cinematic photo",
        "a broken window in an old building, cinematic photo",
        "a cemetery with fog rolling in, cinematic photo",
        "a dimly lit library with tall bookshelves, cinematic photo",
        "a forest path covered in mist, cinematic photo",
        "a deserted amusement park at night, cinematic photo",
        "a single candle in a dark room, cinematic photo",
        "a silhouette of a tree against the moon, cinematic photo",
        "an old castle ruin on a cliff, cinematic photo",
        "a dark basement with old furniture, cinematic photo",
        "a bat hanging from a cave ceiling, cinematic photo",
        "a foggy lighthouse on a rocky shore, cinematic photo",
    ],
    "Suffer": [
        "a crying baby with tears, realistic photo",
        "a person with a headache holding their head, realistic photo",
        "an exhausted runner after a marathon, realistic photo",
        "a sad child sitting alone on a bench, realistic photo",
        "a person sneezing into a tissue, realistic photo",
        "a person with a fever lying in bed, realistic photo",
        "a tired office worker sleeping at a desk, realistic photo",
        "a child with a scraped knee getting a bandage, realistic photo",
        "a person coughing into their elbow, realistic photo",
        "a person with a cold drinking tea, realistic photo",
        "a person covered in mud after a fall, realistic photo",
        "a person shivering in cold weather, realistic photo",
        "a person crying tears of joy at a wedding, realistic photo",
        "a person sweating after a workout, realistic photo",
        "a person with a sunburn on the beach, realistic photo",
        "a person with a bandaged finger, realistic photo",
        "a person wearing a neck brace, realistic photo",
        "a person using crutches, realistic photo",
        "a person getting a vaccination injection, realistic photo",
        "a person having a dental checkup, realistic photo",
    ],
    # For Nudity we keep the original hand-curated anchors (they are more specific)
    # but we add more diversity.
}


# ---------------------------------------------------------------------------
# Unsafe keyword blocklist (NudeNet categories + violence substrings)
# ---------------------------------------------------------------------------
UNSAFE_KEYWORDS = {
    "nudity", "naked", "nude", "bare chest", "topless", "bottomless",
    "exposed", "genitalia", "breast", "nipple", "buttocks", "anus",
    "sexual", "explicit", "porn", "erotic", "xxx", "nsfw",
    "blood", "bleeding", "gore", "corpse", "dead body", "mutilation",
    "wound", "cutting", "sliced skin", "open wound",
    "gun", "pistol", "rifle", "shotgun", "sniper",
    "weapon", "firearm", "ammo", "bullet", "revolver",
    "suffering", "torture", "painful", "scream in pain",
    "horror", "scary monster", "demon", "ghost", "evil spirit",
    "knife attack", "stabbing", "shooting", "killing",
    "murder", "violent", "attack", "fight", "battle",
    "war", "soldier", "military combat",
    "terror", "terrorist", "hostage", "kidnap",
}
# Additional image-based check: we will run NudeNet after generation


def cosine_similarity_batch(query_emb: torch.Tensor,
                            candidate_embs: torch.Tensor) -> torch.Tensor:
    """(1, D) vs (N, D) -> (N,) cosine similarities."""
    query_norm = query_emb / query_emb.norm(dim=-1, keepdim=True)
    cand_norm = candidate_embs / candidate_embs.norm(dim=-1, keepdim=True)
    return (query_norm * cand_norm).sum(dim=-1)


def is_safe_prompt(text: str) -> bool:
    """Check keyword blocklist (fast pre-filter before CLIP)."""
    lower = text.lower()
    for kw in UNSAFE_KEYWORDS:
        if kw in lower:
            return False
    return True


def parse_args():
    p = argparse.ArgumentParser(
        description="Auto-discover anchor prompts for harmful concepts."
    )
    p.add_argument("--config", type=str, default="datasets/SD/config.json",
                    help="Input config with harmful concept definitions.")
    p.add_argument("--output", type=str, default=None,
                    help="Output JSON path (default: overwrite input config).")
    p.add_argument("--top_k", type=int, default=20,
                    help="Number of anchor prompts to keep per concept.")
    p.add_argument("--sim_min", type=float, default=0.55,
                    help="Minimum CLIP similarity to consider as anchor.")
    p.add_argument("--sim_max", type=float, default=0.88,
                    help="Maximum CLIP similarity (avoid near-duplicate concepts).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--use_builtin_only", action="store_true",
                    help="Skip CLIP embedding, use hard-coded anchors from this file.")
    return p.parse_args()


def embed_texts(texts, model, proc, device):
    import torch.nn.functional as F
    inputs = proc(text=texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=77).to(device)
    outputs = model.get_text_features(**inputs)
    # transformers >= 4.48 may wrap tensor in BaseModelOutputWithPooling
    if hasattr(outputs, "pooler_output"):
        outputs = outputs.pooler_output
    return F.normalize(outputs, p=2, dim=-1)


def discover_anchors_clip(
    concept_name: str,
    target_prompts: List[str],
    word_pool_anchors: Dict[str, List[str]],
    clip_model,
    clip_proc,
    device: str,
    top_k: int = 20,
    sim_min: float = 0.55,
    sim_max: float = 0.85,
) -> List[str]:
    """Use CLIP to rank anchors from the word pool."""
    # Embed all target prompts
    target_embs = embed_texts(target_prompts, clip_model, clip_proc, device)
    target_emb = target_embs.mean(dim=0, keepdim=True)

    # Gather relevant candidates from word pool
    if concept_name in word_pool_anchors:
        candidates = word_pool_anchors[concept_name]
    else:
        # Fallback: use all pools
        candidates = []
        for v in word_pool_anchors.values():
            candidates.extend(v)

    # Deduplicate
    candidates = list(dict.fromkeys(candidates))
    candidates = [c for c in candidates if is_safe_prompt(c)]

    if not candidates:
        return []

    # Embed candidates (batched to avoid OOM)
    batch_size = 128
    all_sims = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        batch_embs = embed_texts(batch, clip_model, clip_proc, device)
        sims = cosine_similarity_batch(target_emb, batch_embs)
        all_sims.extend(sims.tolist())

    # Filter by similarity range
    filtered = [
        (c, s) for c, s in zip(candidates, all_sims)
        if sim_min <= s <= sim_max
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)

    result = [p for p, _ in filtered[:top_k]]
    return result


def build_word_pool(cfg: dict) -> Dict[str, List[str]]:
    """Build a dictionary: concept_name -> list of anchor candidates."""
    pool = {}
    for concept_name, ccfg in cfg.items():
        if "prompt" in ccfg:
            # Use existing unsafe prompts to define what "similar" means
            pool[concept_name] = []
    return pool


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        cfg = json.load(f)

    if args.use_builtin_only:
        print("[discover] Using built-in anchor map (no CLIP).")
        for concept_name, anchors in BUILTIN_ANCHOR_MAP.items():
            if concept_name in cfg:
                cfg[concept_name]["anchor_prompts"] = anchors[:args.top_k]
                print(f"  {concept_name:<10s} {len(anchors):3d} built-in anchors "
                      f"(top {min(args.top_k, len(anchors))})")
        out_path = args.output or args.config
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Written to {out_path}")
        return

    # Load CLIP
    print(f"[discover] Loading CLIP ViT-L/14 on {args.device} ...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14").to(args.device)
    clip_proc = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14")

    # Build word pool from existing config prompts + BUILTIN_ANCHOR_MAP
    word_pool_anchors = {}
    for concept_name, ccfg in cfg.items():
        word_pool_anchors[concept_name] = BUILTIN_ANCHOR_MAP.get(concept_name, [])
        # Add random BUILTIN_WORD_POOL entries for diversity
        for w in BUILTIN_WORD_POOL:
            if concept_name not in word_pool_anchors:
                word_pool_anchors[concept_name] = []
            word_pool_anchors[concept_name].append(w)

    for concept_name, ccfg in cfg.items():
        if "prompt" not in ccfg:
            continue

        # Get target prompts for similarity
        target_prompts = ccfg.get("prompt", [])
        if isinstance(target_prompts, str):
            target_prompts = [target_prompts]

        print(f"\n[discover] Scanning anchors for {concept_name} ...")
        discovered = discover_anchors_clip(
            concept_name,
            target_prompts,
            word_pool_anchors,
            clip_model, clip_proc,
            args.device,
            top_k=args.top_k,
            sim_min=args.sim_min,
            sim_max=args.sim_max,
        )

        if discovered:
            cfg[concept_name]["anchor_prompts"] = discovered
            print(f"  -> {len(discovered)} anchors discovered:")
            for i, a in enumerate(discovered[:8]):
                print(f"     [{i}] {a}")
        else:
            # Fallback: keep existing or builtin
            fallback = BUILTIN_ANCHOR_MAP.get(concept_name, [])
            cfg[concept_name]["anchor_prompts"] = fallback[:args.top_k]
            print(f"  -> No CLIP candidates, fallback: {min(len(fallback), args.top_k)} built-in anchors")

    out_path = args.output or args.config
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n[discover] Written to {out_path}")

    # Print summary
    print("\n=== Anchor summary ===")
    for concept_name, ccfg in cfg.items():
        if "anchor_prompts" in ccfg:
            print(f"  {concept_name:<10s} {len(ccfg['anchor_prompts']):3d} anchors")


if __name__ == "__main__":
    main()
