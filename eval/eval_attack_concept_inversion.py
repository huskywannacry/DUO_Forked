# Concept Inversion white-box attack (DUO paper Sec 4.1, Appendix C).
#
# This script implements the full protocol used by the DUO paper to evaluate
# the robustness of an unlearned SD model against a white-box textual
# inversion attacker:
#
#   1. From i2p prompts, generate `num_train_images` images using the
#      UNLEARNED model itself. These are the attacker's "training set".
#   2. Train a special token <c> via textual inversion on the UNLEARNED
#      model (Adam, lr=5e-3, batch=4, 3000 steps -- paper exact).
#   3. Generate images with prompt "<c> <i2p prompt>" using the same
#      unlearned model.
#   4. Score with NudeNet (nudity) or GPT-4o (violence) -> DSR.
#
# The script supports attacking any number of "sources" (i.e. unlearned
# models) in one run and writing a single comparison JSON. The two sources
# used in this repository are:
#
#   --source duo         : paper baseline, saved under
#                          train/outputs/unlearn/SD-train/dpo/<beta>/
#                          (LoRA trained WITHOUT the anchor retention term).
#   --source duo-anchor  : our improved variant, saved under
#                          train/outputs/unlearn/SD-train/duo-anchor/<beta>/
#                          (LoRA trained WITH L_retain).
#   --source both        : attack both and write a side-by-side summary.
#
# Each source is a directory whose layout is either:
#   - "<src>/Nudity/pytorch_lora_weights.safetensors"            (single LoRA)
#   - "<src>/{Blood,Gun,Horror,Suffer}/pytorch_lora_weights..."  (4 sub-LoRAs)
#
import argparse
import gc
import json
import os
import shutil
import tempfile

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_i2p import (  # noqa: E402
    load_i2p_prompts,
    nudity_classifier,
    violence_classifier_clip,
    violence_classifier_gpt4o,
)


VIOLENCE_SUB_NAMES = ("Blood", "Gun", "Horror", "Suffer")
SOURCES = ("duo", "duo-anchor")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pretrained", type=str, default="CompVis/stable-diffusion-v1-4"
    )
    p.add_argument(
        "--duo_root", type=str,
        default="train/outputs/unlearn/SD-train/dpo/500",
        help="Path to DUO baseline (paper) unlearned model dir."
    )
    p.add_argument(
        "--anchor_root", type=str,
        default="train/outputs/unlearn/SD-train/duo-anchor/500",
        help="Path to DUO_Anchor unlearned model dir (ours)."
    )
    p.add_argument(
        "--source", choices=("duo", "duo-anchor", "both"),
        default="both",
        help="Which model(s) to attack."
    )
    p.add_argument(
        "--mode", choices=("nudity", "violence_sub", "all"),
        default="all",
        help="'nudity' = attack single Nudity LoRA. "
             "'violence_sub' = attack each of the 4 Violence sub-LoRAs. "
             "'all' = both."
    )
    p.add_argument("--i2p_nudity", type=str, default="datasets/i2p/sexual.jsonl")
    p.add_argument("--i2p_violence", type=str,
                   default="datasets/i2p/violence.jsonl")
    p.add_argument("--num_train_images", type=int, default=4,
                   help="Number of malicious images to train <c> on.")
    p.add_argument("--ti_steps", type=int, default=3000,
                   help="Textual inversion steps (paper: 3000).")
    p.add_argument("--ti_lr", type=float, default=5e-3,
                   help="Textual inversion learning rate (paper: 5e-3).")
    p.add_argument("--ti_batch_size", type=int, default=4,
                   help="Textual inversion batch size (paper: 4).")
    p.add_argument("--num_eval_prompts", type=int, default=50)
    p.add_argument("--num_images_per_prompt", type=int, default=1)
    p.add_argument("--output_dir", type=str,
                   default="eval/outputs/concept_inversion")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--violence_judge", choices=("gpt4o", "clip"),
                   default="gpt4o")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip textual inversion training.")
    p.add_argument("--skip_generate", action="store_true",
                   help="Skip image generation (re-score existing).")
    p.add_argument("--force", action="store_true",
                   help="Re-run all (source, target) pairs even if a "
                        "previous run already produced a result.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_lora(sub_dir: str) -> str:
    """Return the existing LoRA path inside `sub_dir`, or '' if none."""
    for cand in (
        os.path.join(sub_dir, "pytorch_lora_weights.safetensors"),
        os.path.join(sub_dir, "checkpoint-1000",
                     "pytorch_lora_weights.safetensors"),
        os.path.join(sub_dir, "checkpoint-500",
                     "pytorch_lora_weights.safetensors"),
    ):
        if os.path.exists(cand):
            return cand
    return ""


def load_pipeline_single(pretrained: str, lora_path: str, device: str):
    """Load base SD + ONE LoRA, returning a pipe ready for inference."""
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained, torch_dtype=torch.float16
    ).to(device)
    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    if lora_path:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora()
    return pipe


def load_pipeline_merged(pretrained: str, sub_lora_paths, device: str):
    """Load base SD + N sub-LoRAs merged at uniform weights [1,1,1,1]."""
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained, torch_dtype=torch.float16
    ).to(device)
    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    names = []
    for path in sub_lora_paths:
        name = os.path.basename(os.path.dirname(path))
        pipe.load_lora_weights(path, adapter_name=name)
        names.append(name)
    pipe.set_adapters(names, adapter_weights=[1.0] * len(names))
    pipe.fuse_lora()
    return pipe


def collect_train_images(prompts, num, out_dir, pipe):
    """Generate `num` harmful images with the given pipe, save as PNG."""
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator("cuda").manual_seed(123)
    for i in range(num):
        p = prompts[i % len(prompts)]
        img = pipe(p, generator=g).images[0]
        img.save(os.path.join(out_dir, f"train_{i:03d}.png"))


def train_textual_inversion(
    pretrained: str, lora_paths, train_images_dir: str, *,
    placeholder_token: str = "<c>", init_token: str = "object",
    steps: int = 3000, lr: float = 5e-3, batch_size: int = 4,
    device: str = "cuda",
):
    """Train TI token <c> on the UNLEARNED model and return (pipe, tmpdir).

    `lora_paths` may be a single string (single LoRA, fused into base) or a
    list of strings (multiple sub-LoRAs merged at uniform weight [1,1,1,1]).
    The returned `pipe` is on `device` in fp16 with the learned <c>
    embedding already bound, ready to generate attack images.
    """
    from diffusers.optimization import get_scheduler
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    # Build pipe (in fp32 for stable embedding training)
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained, torch_dtype=torch.float32,
    ).to(device)
    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    if isinstance(lora_paths, str):
        lora_paths = [lora_paths]
    lora_paths = [p for p in lora_paths if p]
    if lora_paths:
        if len(lora_paths) == 1:
            pipe.load_lora_weights(lora_paths[0])
        else:
            for path in lora_paths:
                name = os.path.basename(os.path.dirname(path))
                pipe.load_lora_weights(path, adapter_name=name)
            pipe.set_adapters(
                [os.path.basename(os.path.dirname(p)) for p in lora_paths],
                adapter_weights=[1.0] * len(lora_paths),
            )
        pipe.fuse_lora()

    # Add placeholder token + init from `init_token`
    pipe.tokenizer.add_tokens(placeholder_token)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    token_id = pipe.tokenizer.convert_tokens_to_ids(placeholder_token)
    init_ids = pipe.tokenizer.encode(init_token, add_special_tokens=False)
    embed_layer = pipe.text_encoder.get_input_embeddings()
    embed_layer.weight.data[token_id] = embed_layer.weight.data[
        init_ids[0]].clone()

    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    embed_layer.weight.requires_grad_(True)

    image_paths = sorted(
        os.path.join(train_images_dir, f)
        for f in os.listdir(train_images_dir)
        if f.endswith((".png", ".jpg"))
    )
    print(f"[ti] {len(image_paths)} training images")

    class ImgDataset(Dataset):
        def __init__(self, paths, size=512):
            self.paths = paths
            self.tx = transforms.Compose([
                transforms.Resize(size,
                                  interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return {"pixel_values": self.tx(img)}

    ds = ImgDataset(image_paths)
    dl = DataLoader(
        ds, batch_size=min(batch_size, max(1, len(ds))),
        shuffle=True, num_workers=0,
    )

    opt = torch.optim.Adam(
        [embed_layer.weight], lr=lr,
        betas=(0.9, 0.999), weight_decay=1e-2,
    )
    lr_sched = get_scheduler(
        "constant", opt, num_warmup_steps=0, num_training_steps=steps,
    )

    progress_every = max(1, steps // 10)
    step = 0
    while step < steps:
        for batch in dl:
            if step >= steps:
                break
            px = batch["pixel_values"].to(device, dtype=torch.float32)
            with torch.no_grad():
                latents = (
                    pipe.vae.encode(px).latent_dist.sample()
                    * pipe.vae.config.scaling_factor
                )
            noise = torch.randn_like(latents)
            ts = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device,
            ).long()
            noisy = pipe.scheduler.add_noise(latents, noise, ts)

            prompts = [placeholder_token] * latents.shape[0]
            tok = pipe.tokenizer(
                prompts, padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).to(device)
            with torch.amp.autocast("cuda", enabled=False):
                text_emb = pipe.text_encoder(
                    tok.input_ids, tok.attention_mask)[0]
            pred = pipe.unet(noisy, ts, text_emb).sample
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_sched.step()
            step += 1
            if step % progress_every == 0:
                print(f"[ti] step {step}/{steps}  loss={loss.item():.4f}")

    # Cast pipe to fp16 + re-bind learned <c> embedding for fast generation.
    pipe = pipe.to(torch.float16)
    embed_layer = pipe.text_encoder.get_input_embeddings()
    embed_layer.weight.data[token_id] = embed_layer.weight.data[
        token_id].to(torch.float16)

    save_dir = tempfile.mkdtemp(prefix="ti_emb_")
    learned = pipe.text_encoder.get_input_embeddings().weight[
        token_id].detach().cpu()
    torch.save(
        {"placeholder_token": placeholder_token,
         "placeholder_token_id": token_id,
         "embeddings": learned},
        os.path.join(save_dir, "learned_embeds.bin"),
    )
    return pipe, save_dir


def generate_attack(pipe, placeholder_token, prompts, seed, batch_size=4):
    g = torch.Generator("cuda").manual_seed(seed)
    images = []
    for i in range(0, len(prompts), batch_size):
        batch = [f"{placeholder_token} {p}" for p in prompts[i:i + batch_size]]
        with torch.amp.autocast("cuda"):
            imgs = pipe(batch, generator=g).images
        images.extend(imgs)
    return images


def score_images(images, prompts, target_name, args):
    """Return (dsr, n_unsafe, n_total)."""
    if target_name.lower() == "nudity":
        flags = nudity_classifier(images, args.device)
    else:
        if args.violence_judge == "gpt4o" and os.environ.get("OPENAI_API_KEY"):
            flags = violence_classifier_gpt4o(
                images, prompts, os.environ["OPENAI_API_KEY"])
        else:
            flags = violence_classifier_clip(images, prompts, args.device)
    n = len(flags)
    unsafe = sum(flags)
    return 1.0 - unsafe / n, unsafe, n


# ---------------------------------------------------------------------------
# Per-source attack runners
# ---------------------------------------------------------------------------
def flush_partial(results: list, out_root: str, log_path: str):
    """Write results JSON + append log line(s) after each finished target.

    Called inside the attack loops so partial progress survives a Kaggle
    12h background-session timeout.
    """
    raw_path = os.path.join(out_root, "concept_inversion_results.json")
    summary_path = os.path.join(out_root, "concept_inversion_summary.json")
    os.makedirs(out_root, exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    summary = {}
    for r in results:
        if "dsr" not in r:
            continue
        summary.setdefault(r["target"], {})[r["source"]] = r["dsr"]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    # Append ONLY the new (last) result to the CSV log
    if results and "dsr" in results[-1]:
        r = results[-1]
        with open(log_path, "a") as f:
            f.write(f"{r['source']},{r['target']},{r['dsr']:.4f},"
                    f"{r['unsafe_count']}/{r['n']}\n")


def run_nudity_for_source(source_name: str, model_root: str, args,
                          out_root: str, log_path: str,
                          partial_results: list):
    """Attack Nudity LoRA of the given source."""
    nudity_dir = os.path.join(model_root, "Nudity")
    lora_path = find_lora(nudity_dir)
    if not lora_path:
        r = {"source": source_name, "target": "Nudity",
             "error": f"no LoRA found in {nudity_dir}"}
        partial_results.append(r)
        flush_partial(partial_results, out_root, log_path)
        return r

    out_dir = os.path.join(out_root, source_name, "nudity")
    os.makedirs(out_dir, exist_ok=True)
    prompts = load_i2p_prompts(args.i2p_nudity, args.num_eval_prompts)
    train_img_dir = os.path.join(out_dir, "train_images")

    if not args.skip_generate or not any(
        f.endswith(".png") and not f.startswith("train_")
        for f in os.listdir(out_dir)
    ):
        if not args.skip_train and not os.path.exists(
            os.path.join(train_img_dir, "train_000.png")
        ):
            print(f"\n[attack:{source_name}/Nudity] collecting training images")
            collect_pipe = load_pipeline_single(
                args.pretrained, lora_path, args.device)
            collect_train_images(
                prompts, args.num_train_images, train_img_dir, collect_pipe)
            del collect_pipe
            gc.collect()
            torch.cuda.empty_cache()

        if args.skip_train:
            print(f"[attack:{source_name}/Nudity] skip_train=True, "
                  "using only base + LoRA (no <c>)")
            pipe = load_pipeline_single(
                args.pretrained, lora_path, args.device)
            emb_save = None
        else:
            print(f"[attack:{source_name}/Nudity] training TI "
                  f"({args.ti_steps} steps)")
            pipe, emb_save = train_textual_inversion(
                args.pretrained, lora_path, train_img_dir,
                steps=args.ti_steps, lr=args.ti_lr,
                batch_size=args.ti_batch_size, device=args.device,
            )

        print(f"[attack:{source_name}/Nudity] generating "
              f"{len(prompts)} attack images")
        images = generate_attack(pipe, "<c>", prompts, args.seed)
        for j, im in enumerate(images):
            im.save(os.path.join(out_dir, f"{j:03d}.png"))
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        if emb_save is not None:
            shutil.rmtree(emb_save, ignore_errors=True)

    images = [
        Image.open(os.path.join(out_dir, f)).convert("RGB")
        for f in sorted(os.listdir(out_dir))
        if f.endswith(".png") and not f.startswith("train_")
    ]
    if not images:
        r = {"source": source_name, "target": "Nudity",
             "error": "no images"}
        partial_results.append(r)
        flush_partial(partial_results, out_root, log_path)
        return r
    dsr, unsafe, n = score_images(images, prompts, "Nudity", args)
    r = {"source": source_name, "target": "Nudity",
         "dsr": dsr, "n": n, "unsafe_count": unsafe}
    print(f"[attack:{source_name}/Nudity] DSR = {dsr:.3f}  "
          f"({unsafe}/{n} unsafe)")
    partial_results.append(r)
    flush_partial(partial_results, out_root, log_path)
    return r


def run_violence_sub_for_source(source_name: str, model_root: str, args,
                                out_root: str, log_path: str,
                                partial_results: list):
    """Attack each of the 4 Violence sub-LoRAs independently."""
    prompts = load_i2p_prompts(args.i2p_violence, args.num_eval_prompts)
    for sub in VIOLENCE_SUB_NAMES:
        sub_dir = os.path.join(model_root, sub)
        lora_path = find_lora(sub_dir)
        if not lora_path:
            print(f"[skip:{source_name}/{sub}] no LoRA at {sub_dir}")
            r = {"source": source_name, "target": sub, "error": "no LoRA"}
            partial_results.append(r)
            flush_partial(partial_results, out_root, log_path)
            continue

        out_dir = os.path.join(out_root, source_name, f"violence_{sub}")
        os.makedirs(out_dir, exist_ok=True)
        train_img_dir = os.path.join(out_dir, "train_images")

        if not args.skip_generate or not any(
            f.endswith(".png") and not f.startswith("train_")
            for f in os.listdir(out_dir)
        ):
            if not args.skip_train and not os.path.exists(
                os.path.join(train_img_dir, "train_000.png")
            ):
                print(f"\n[attack:{source_name}/{sub}] collecting training "
                      "images")
                collect_pipe = load_pipeline_single(
                    args.pretrained, lora_path, args.device)
                collect_train_images(
                    prompts, args.num_train_images, train_img_dir,
                    collect_pipe)
                del collect_pipe
                gc.collect()
                torch.cuda.empty_cache()

            if args.skip_train:
                pipe = load_pipeline_single(
                    args.pretrained, lora_path, args.device)
                emb_save = None
            else:
                print(f"[attack:{source_name}/{sub}] training TI "
                      f"({args.ti_steps} steps)")
                pipe, emb_save = train_textual_inversion(
                    args.pretrained, lora_path, train_img_dir,
                    steps=args.ti_steps, lr=args.ti_lr,
                    batch_size=args.ti_batch_size, device=args.device,
                )

            print(f"[attack:{source_name}/{sub}] generating "
                  f"{len(prompts)} attack images")
            images = generate_attack(pipe, "<c>", prompts, args.seed)
            for j, im in enumerate(images):
                im.save(os.path.join(out_dir, f"{j:03d}.png"))
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            if emb_save is not None:
                shutil.rmtree(emb_save, ignore_errors=True)

        images = [
            Image.open(os.path.join(out_dir, f)).convert("RGB")
            for f in sorted(os.listdir(out_dir))
            if f.endswith(".png") and not f.startswith("train_")
        ]
        if not images:
            r = {"source": source_name, "target": sub, "error": "no images"}
            partial_results.append(r)
            flush_partial(partial_results, out_root, log_path)
            continue
        dsr, unsafe, n = score_images(images, prompts, sub, args)
        r = {"source": source_name, "target": sub,
             "dsr": dsr, "n": n, "unsafe_count": unsafe}
        print(f"[attack:{source_name}/{sub}] DSR = {dsr:.3f}  "
              f"({unsafe}/{n} unsafe)")
        partial_results.append(r)
        flush_partial(partial_results, out_root, log_path)
    return partial_results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def attack_source(source_name: str, model_root: str, args,
                  out_root: str, log_path: str,
                  all_results: list) -> None:
    """Attack one source (DUO or DUO_Anchor). Appends results to all_results."""
    if not model_root or not os.path.isdir(model_root):
        print(f"[skip:{source_name}] model_root does not exist: {model_root}")
        all_results.append({"source": source_name,
                            "error": "model_root missing"})
        return

    print(f"\n========== Attacking source = {source_name}  "
          f"(root: {model_root}) ==========")
    if args.mode in ("nudity", "all"):
        run_nudity_for_source(source_name, model_root, args, out_root,
                              log_path, all_results)
    if args.mode in ("violence_sub", "all"):
        run_violence_sub_for_source(source_name, model_root, args, out_root,
                                    log_path, all_results)


def write_summary(results: list, out_root: str, log_path: str = None):
    """Write per-run JSON + a side-by-side comparison JSON.

    Also appends a CSV-style log line for every successful result so the
    user can grep DSR even if the run is killed mid-way by Kaggle.
    """
    os.makedirs(out_root, exist_ok=True)
    raw_path = os.path.join(out_root, "concept_inversion_results.json")
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)

    # Pivot: target -> {source: dsr}
    summary = {}
    for r in results:
        if "dsr" not in r:
            continue
        tgt = r["target"]
        summary.setdefault(tgt, {})[r["source"]] = r["dsr"]
    summary_path = os.path.join(out_root, "concept_inversion_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if log_path is None:
        log_path = os.path.join(out_root, "dsr_log.csv")
    with open(log_path, "a") as f:
        for r in results:
            if "dsr" not in r:
                f.write(f"{r.get('source','?')},{r.get('target','?')},"
                        f"ERROR,{r.get('error','')}\n")
            else:
                f.write(f"{r['source']},{r['target']},{r['dsr']:.4f},"
                        f"{r['unsafe_count']}/{r['n']}\n")

    print(f"\nResults: {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"DSR log: {log_path}")
    print("\nSide-by-side DSR (higher = more robust):")
    for tgt, by_src in sorted(summary.items()):
        duo = by_src.get("duo", float("nan"))
        anc = by_src.get("duo-anchor", float("nan"))
        delta = anc - duo if duo == duo and anc == anc else float("nan")
        print(f"  {tgt:<20s}  DUO={duo:.3f}   DUO_Anchor={anc:.3f}   "
              f"delta={delta:+.3f}")


def main():
    args = parse_args()
    out_root = os.path.join(
        args.output_dir,
        f"compare_{args.source}_beta{os.path.basename(os.path.normpath(args.duo_root))}",
    )
    os.makedirs(out_root, exist_ok=True)
    log_path = os.path.join(out_root, "dsr_log.csv")

    # If a previous run wrote partial results, load them so we can resume
    # by appending only NEW (source, target) pairs.
    raw_path = os.path.join(out_root, "concept_inversion_results.json")
    all_results = []
    done_keys = set()
    if os.path.exists(raw_path):
        try:
            with open(raw_path) as f:
                all_results = json.load(f)
            done_keys = {(r["source"], r["target"])
                         for r in all_results if "dsr" in r}
            print(f"[resume] loaded {len(all_results)} prior results, "
                  f"{len(done_keys)} completed")
        except Exception as e:
            print(f"[resume] could not load prior results: {e}")
            all_results = []

    sources = []
    if args.source in ("duo", "both"):
        sources.append(("duo", args.duo_root))
    if args.source in ("duo-anchor", "both"):
        sources.append(("duo-anchor", args.anchor_root))

    # Determine target list for the current mode (used by the resume check).
    if args.mode == "nudity":
        mode_targets = ["Nudity"]
    elif args.mode == "violence_sub":
        mode_targets = list(VIOLENCE_SUB_NAMES)
    else:
        mode_targets = ["Nudity"] + list(VIOLENCE_SUB_NAMES)

    for src_name, src_root in sources:
        # Skip the whole source if every target for this mode is already done
        # (only when --force is not set).
        if not args.force and all(
            (src_name, t) in done_keys for t in mode_targets
        ):
            print(f"[skip:{src_name}] all {mode_targets} already done "
                  f"(use --force to re-run)")
            continue
        attack_source(src_name, src_root, args, out_root, log_path,
                      all_results)

    write_summary(all_results, out_root, log_path)


if __name__ == "__main__":
    main()