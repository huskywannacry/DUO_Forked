# DUO-Anchor v3: End-to-end Kaggle notebook với Paper-grade Evaluation
#
# Pipeline gồm:
#   1. Clone repo + cài deps
#   2. Dataset + Enhanced anchors
#   3. Train DUO baseline + DUO_Anchor (10 LoRAs)
#   4. Concept Inversion attack  → DSR bảng
#   5. Ring-A-Bell attack        → DSR bảng
#   6. FID + CLIP on MS COCO 10k → (MỚI: paper Sec 4.1 — COCO thật)
#   7. LPIPS Anchor Retention
#   8. Pack + zip kết quả
#
# Walltime ước tính (Kaggle P100, 1 GPU):
#   Setup + data                ~20 min
#   Enhanced anchor discovery   ~15 min
#   Train DUO baseline          ~75 min
#   Train DUO_Anchor            ~98 min
#   Concept Inversion attack    ~2.5 h
#   Ring-A-Bell attack          ~50 min
#   FID/CLIP COCO 10k           ~3 h     (MỚI: paper Sec 4.1)
#   LPIPS + misc                ~15 min
#   TOTAL                       ~9 h     (fit 12h background)
#
# KAGGLE SETTINGS
#   * Accelerator:  GPU P100 (hoặc T4 x2)
#   * Internet:     ON (cần tải COCO + HuggingFace datasets)
#   * Background:   ON (Save Version -> Run All -> "Save as background")
#   * Optional:     Add-ons > Secrets > add OPENAI_API_KEY

# ============================================================
# CELL 1: CLONE REPO + INSTALL
# ============================================================
import os
os.environ["WANDB_MODE"] = "offline"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

REPO = "https://github.com/huskywannacry/DUO_Forked.git"
BRANCH = "main"

if not os.path.exists("/kaggle/working/DUO-Anchor"):
    !git clone -b {BRANCH} {REPO} /kaggle/working/DUO-Anchor
%cd /kaggle/working/DUO-Anchor

# Patch model_root path trong attack scripts
!sed -i 's|save_dir="./outputs"|save_dir="./train/outputs"|g' \
    scripts/attack-both-all.sh scripts/attack-ring-a-bell-all.sh

!pip install -q -r requirements.txt
!pip install -q 'diffusers<=0.29.0' 'peft<=0.12.0' 'transformers<=4.44.0' 'accelerate<=1.0.0'
!pip install -q git+https://github.com/notAI-tech/NudeNet.git

# Cài thêm datasets + torchmetrics cho COCO 30k FID/CLIP
!pip install -q datasets torchmetrics pytorch-fid

import torch
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, "
      f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# ============================================================
# CELL 2: load OPENAI_API_KEY
# ============================================================
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    secret_names = [s.name for s in secrets.list_secrets()] if hasattr(secrets, "list_secrets") else []
    if "OPENAI_API_KEY" in secret_names:
        os.environ["OPENAI_API_KEY"] = secrets.get_secret("OPENAI_API_KEY")
        print("OPENAI_API_KEY loaded")
    else:
        print("OPENAI_API_KEY not in secrets (violence will use CLIP proxy)")
except Exception as e:
    print(f"no Kaggle secrets access: {e}")


# ============================================================
# CELL 3a: DATASET
# ============================================================
KAGGLE_DATASET_SD = "/kaggle/input/datasets/kientheconqueror/duo-anchor/DUO-Anchor/datasets/SD"
import os, subprocess, pathlib

TARGET_SD = "datasets/SD"

if os.path.exists(KAGGLE_DATASET_SD):
    print("Found existing dataset in Kaggle Input. Copying safe/unsafe data...")
    subprocess.run(f"rm -rf {TARGET_SD}", shell=True)
    subprocess.run(f"cp -r {KAGGLE_DATASET_SD} {TARGET_SD}", shell=True)
else:
    print("No input dataset. Generating safe/unsafe from scratch (~25 min)...")
    !bash scripts/prepare-dataset.sh

print("Generating fresh anchor images...")
!bash scripts/prepare-anchor.sh

!echo "=== datasets/SD ==="
!ls datasets/SD/
!echo "=== basic anchor images per concept ==="
for c in ["Nudity", "Blood", "Gun", "Horror", "Suffer"]:
    result = subprocess.run(
        f"ls datasets/SD/{c}/anchor/ 2>/dev/null | wc -l",
        shell=True, capture_output=True, text=True
    )
    count = result.stdout.strip()
    print(f"  {c:<8s}  {count if count else '0'} basic anchor images")

!echo "zipping datasets/SD for reuse ..."
!cd /kaggle/working && tar -czf DUO-Anchor-datasets.tar.gz DUO-Anchor/datasets/SD/ 2>/dev/null


# ============================================================
# CELL 3b: ENHANCED ANCHOR AUTO-DISCOVERY (~15 min)
# ============================================================
print("=" * 60)
print("ENHANCED ANCHOR: Step 1 — Auto-discover prompts (CLIP text space)")
print("=" * 60)
!python3 datasets/SD/discover_anchors.py \
    --config       datasets/SD/config.json \
    --output       datasets/SD/config_auto_anchor.json \
    --top_k        20 \
    --sim_min      0.55 \
    --sim_max      0.88 \
    --device       "cuda:0" 2>&1

print("")
print("=" * 60)
print("ENHANCED ANCHOR: Step 2 — Generate + filter images (CLIP + NudeNet)")
print("=" * 60)
!python3 datasets/SD/generate_anchors_enhanced.py \
    --config                    datasets/SD/config_auto_anchor.json \
    --data_dir                  datasets/SD \
    --device                    "cuda:0" \
    --per_prompt                6 \
    --max_anchors               24 \
    --sim_min                   0.40 \
    --sim_max                   0.90 \
    --diversity_threshold       0.97 \
    --nudenet_filter 2>&1

!echo "=== enhanced anchor images per concept ==="
for c in ["Nudity", "Blood", "Gun", "Horror", "Suffer"]:
    result = subprocess.run(
        f"ls datasets/SD/{c}/anchor/ 2>/dev/null | grep -c '\\.jpg'",
        shell=True, capture_output=True, text=True
    )
    count = result.stdout.strip()
    print(f"  {c:<8s}  {count if count else '0'} enhanced anchor images")


# ============================================================
# CELL 4: TRAIN DUO BASELINE (~75 phút)
# ============================================================
!bash scripts/sd-nudity.sh
!bash scripts/sd-violence.sh


# ============================================================
# CELL 5: TRAIN DUO + ANCHOR (~98 phút)
# ============================================================
subprocess.run("cp scripts/sd-nudity-anchor.sh scripts/sd-nudity-anchor-enhanced.sh", shell=True, check=True)
subprocess.run("cp scripts/sd-violence-anchor.sh scripts/sd-violence-anchor-enhanced.sh", shell=True, check=True)
subprocess.run("sed -i 's|config\\.json|config_auto_anchor.json|g' scripts/sd-nudity-anchor-enhanced.sh", shell=True, check=True)
subprocess.run("sed -i 's|config\\.json|config_auto_anchor.json|g' scripts/sd-violence-anchor-enhanced.sh", shell=True, check=True)

subprocess.run("bash scripts/sd-nudity-anchor-enhanced.sh", shell=True, check=True)
subprocess.run("bash scripts/sd-violence-anchor-enhanced.sh", shell=True, check=True)


# ============================================================
# CELL 6: VERIFY CHECKPOINTS
# ============================================================
exp_root = pathlib.Path("train/outputs/unlearn/SD-train")
expected = []
for method, beta_targets in [
    ("dpo",        [("500", "Nudity"), ("1000", "Blood"), ("1000", "Gun"),
                    ("1000", "Horror"), ("1000", "Suffer")]),
    ("duo-anchor", [("500", "Nudity"), ("1000", "Blood"), ("1000", "Gun"),
                    ("1000", "Horror"), ("1000", "Suffer")]),
]:
    for beta, concept in beta_targets:
        ckpt = exp_root / method / beta / concept / "pytorch_lora_weights.safetensors"
        ok = ckpt.exists()
        expected.append((method, beta, concept, ok))
        print(f"  {'OK' if ok else 'MISSING':<8s}  {method}/{beta}/{concept}")

if not all(ok for *_, ok in expected):
    print("\nWARN: some checkpoints missing. Attack will skip them.")
else:
    print("\nAll 10 LoRA checkpoints present.")


# ============================================================
# CELL 7: CONCEPT INVERSION ATTACK (~2.5h)
# ============================================================
!bash scripts/attack-both-all.sh 2>&1 | tee /kaggle/working/attack-both-all.log


# ============================================================
# CELL 8: AGGREGATE CONCEPT INVERSION RESULTS
# ============================================================
import json, pathlib
out_root = pathlib.Path("eval/outputs/concept_inversion")

merged = {}
for summary_path in out_root.rglob("concept_inversion_summary.json"):
    print(f"loading {summary_path}")
    with open(summary_path) as f:
        s = json.load(f)
    for tgt, by_src in s.items():
        merged.setdefault(tgt, {}).update(by_src)

print("\n" + "=" * 70)
print("TABLE 1: DSR AFTER CONCEPT INVERSION ATTACK (white-box)")
print(" higher DSR = defense more robust")
print("=" * 70)
print(f"{'Concept':<12s}  {'DUO':>10s}  {'DUO_Anchor':>12s}  {'delta':>10s}")
print("-" * 48)
for tgt in ("Nudity", "Blood", "Gun", "Horror", "Suffer"):
    row = merged.get(tgt, {})
    duo = row.get("duo", float("nan"))
    anc = row.get("duo-anchor", float("nan"))
    delta = anc - duo if duo == duo and anc == anc else float("nan")
    print(f"{tgt:<12s}  {duo:>10.3f}  {anc:>12.3f}  {delta:>+10.3f}")

with open(out_root / "FINAL_DSR_TABLE.json", "w") as f:
    json.dump(merged, f, indent=2)


# ============================================================
# CELL 9: RING-A-BELL ATTACK (~50 min)
# ============================================================
print("=" * 60)
print("RING-A-BELL: Nudity + Violence sub-LoRAs (cả 2 sources)")
print("=" * 60)
!bash scripts/attack-ring-a-bell-all.sh 2>&1 | tee /kaggle/working/attack-ring-a-bell-all.log


# ============================================================
# CELL 10: AGGREGATE RING-A-BELL RESULTS
# ============================================================
rab_root = pathlib.Path("eval/outputs/ring_a_bell")
rab_merged = {}
for summary_path in rab_root.rglob("ring_a_bell_summary.json"):
    print(f"loading {summary_path}")
    with open(summary_path) as f:
        s = json.load(f)
    for tgt, by_src in s.items():
        rab_merged.setdefault(tgt, {}).update(by_src)

if rab_merged:
    print("\n" + "=" * 70)
    print("TABLE 2: DSR AFTER RING-A-BELL ATTACK (black-box, GA)")
    print("=" * 70)
    print(f"{'Concept':<12s}  {'DUO':>10s}  {'DUO_Anchor':>12s}  {'delta':>10s}")
    print("-" * 48)
    for tgt in ("Nudity", "Blood", "Gun", "Horror", "Suffer"):
        row = rab_merged.get(tgt, {})
        duo = row.get("duo", float("nan"))
        anc = row.get("duo-anchor", float("nan"))
        delta = anc - duo if duo == duo and anc == anc else float("nan")
        print(f"{tgt:<12s}  {duo:>10.3f}  {anc:>12.3f}  {delta:>+10.3f}")

    with open(rab_root / "FINAL_RING_A_BELL_TABLE.json", "w") as f:
        json.dump(rab_merged, f, indent=2)
else:
    print("\nNo Ring-A-Bell summary files found.")


# ============================================================
# CELL 11: FID + CLIP SCORE on MS COCO 10k (~3 h)  [MỚI]
# ============================================================
# Sử dụng MS COCO 2014 validation captions thật (10k captions)
# qua HuggingFace datasets. Sinh 10k ảnh từ SD1.4 (reference)
# rồi so sánh với ảnh từ unlearned model.
#
# Lưu ý: cell này chạy lâu (~3h). Nếu hết giờ, chạy lại cell này
# trên cùng session để resume (output cached trong output_images_dir).
print("=" * 70)
print("FID + CLIP SCORE on MS COCO 10k (DUO paper Sec 4.1)")
print("=" * 70)

fid_out = "eval/outputs/fid_clip"
os.makedirs(fid_out, exist_ok=True)

# ---- Step 1: Reference (SD1.4 prior) ----
# Chỉ gen nếu chưa có
ref_dir = f"{fid_out}/images/ref"
if not os.path.isdir(ref_dir) or len(os.listdir(ref_dir)) < 100:
    print("\n--- Generating reference SD1.4 images (30k) ---")
    # Dùng subset nhỏ hơn nếu muốn tiết kiệm thời gian
    # Paper gốc dùng 30k, ta dùng 10k để fit 12h
    !python3 eval/eval_fid_clip_coco.py \
        --model_root "train/outputs/unlearn/SD-train/dpo/500" \
        --output "{fid_out}/fid_clip_reference.json" \
        --coco_subset 10000 \
        --gen_batch_size 8 \
        --seed 42 \
        --output_images_dir "{fid_out}/images" 2>&1 | tee "{fid_out}/ref_gen.log"
else:
    print(f"Reference images already exist at {ref_dir}, skipping gen.")

# ---- Step 2: DUO baseline ----
unlearn_dir_duo = f"{fid_out}/images/unlearn_duo"
if not os.path.isdir(unlearn_dir_duo) or len(os.listdir(unlearn_dir_duo)) < 100:
    print("\n--- Generating DUO baseline images (10k) ---")
    !python3 eval/eval_fid_clip_coco.py \
        --model_root "train/outputs/unlearn/SD-train/dpo/500" \
        --output "{fid_out}/fid_clip_duo.json" \
        --coco_subset 10000 \
        --gen_batch_size 8 \
        --seed 42 \
        --output_images_dir "{fid_out}/images" \
        --eval_only 2>&1 | tee "{fid_out}/duo_gen.log"
else:
    print(f"DUO images already exist at {unlearn_dir_duo}, skipping gen.")

# ---- Step 3: DUO-Anchor ----
unlearn_dir_anchor = f"{fid_out}/images/unlearn_anchor"
if not os.path.isdir(unlearn_dir_anchor) or len(os.listdir(unlearn_dir_anchor)) < 100:
    print("\n--- Generating DUO-Anchor images (10k) ---")
    !python3 eval/eval_fid_clip_coco.py \
        --model_root "train/outputs/unlearn/SD-train/duo-anchor/500" \
        --output "{fid_out}/fid_clip_anchor.json" \
        --coco_subset 10000 \
        --gen_batch_size 8 \
        --seed 42 \
        --output_images_dir "{fid_out}/images" \
        --eval_only 2>&1 | tee "{fid_out}/anchor_gen.log"
else:
    print(f"DUO-Anchor images already exist at {unlearn_dir_anchor}, skipping gen.")

# ---- Step 4: Compute FID between reference and each model ----
print("\n=== Computing FID between Reference and Unlearned Models ===")

def compute_fid(ref_path, unlearn_path, label):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchvision import transforms as T
        from PIL import Image

        transform = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        ref_files = sorted([f for f in os.listdir(ref_path) if f.endswith(('.png','.jpg'))])
        unlearn_files = sorted([f for f in os.listdir(unlearn_path) if f.endswith(('.png','.jpg'))])
        n = min(len(ref_files), len(unlearn_files))
        ref_files = ref_files[:n]
        unlearn_files = unlearn_files[:n]
        print(f"  FID ({label}): {n} images per set")

        fid = FrechetInceptionDistance(feature=2048).cuda()
        bs = 32

        for i in range(0, n, bs):
            batch = torch.stack([
                transform(Image.open(os.path.join(ref_path, f)).convert('RGB'))
                for f in ref_files[i:i+bs]
            ]).cuda()
            fid.update(batch, real=True)

        for i in range(0, n, bs):
            batch = torch.stack([
                transform(Image.open(os.path.join(unlearn_path, f)).convert('RGB'))
                for f in unlearn_files[i:i+bs]
            ]).cuda()
            fid.update(batch, real=False)

        return fid.compute().item()
    except Exception as e:
        print(f"  FID error: {e}")
        return None

# Compute FID
fid_duo = None
fid_anchor = None
if os.path.isdir(ref_dir) and os.path.isdir(unlearn_dir_duo):
    fid_duo = compute_fid(ref_dir, unlearn_dir_duo, "DUO")
    print(f"    FID (DUO baseline vs SD1.4): {fid_duo:.4f}" if fid_duo else "    FID: N/A")
else:
    print("  Skipping FID: ref or unlearn images not found.")

if os.path.isdir(ref_dir) and os.path.isdir(unlearn_dir_anchor):
    fid_anchor = compute_fid(ref_dir, unlearn_dir_anchor, "DUO-Anchor")
    print(f"    FID (DUO_Anchor vs SD1.4): {fid_anchor:.4f}" if fid_anchor else "    FID: N/A")

# ---- Step 5: Aggregate ----
print("\n" + "=" * 70)
print("TABLE 4: FID + CLIP SCORE on COCO 30k")
print("=" * 70)
print(f"{'Model':<20s}  {'CLIP Score':>12s}  {'FID':>10s}")
print("-" * 46)

fid_results = {}
for json_path in sorted(pathlib.Path(fid_out).rglob("fid_clip_*.json")):
    with open(json_path) as f:
        data = json.load(f)
    label = json_path.stem.replace("fid_clip_", "")
    cs = data.get("clip_score_unlearn", "N/A")
    f = data.get("fid", "N/A")
    cs_str = f"{cs:.4f}" if isinstance(cs, (int, float)) else str(cs)
    f_str = f"{f:.4f}" if isinstance(f, (int, float)) else str(f)
    fid_results[label] = {"clip_score": cs, "fid": f}
    print(f"  {label:<20s}  {cs_str:>12s}  {f_str:>10s}")

# Also add FID from manual computation (if any)
fid_results["duo_manual_fid"] = {"clip_score": None, "fid": fid_duo}
fid_results["anchor_manual_fid"] = {"clip_score": None, "fid": fid_anchor}

fid_table_path = f"{fid_out}/FID_CLIP_TABLE.json"
with open(fid_table_path, "w") as f:
    json.dump(fid_results, f, indent=2)
print(f"\nFID/CLIP table saved to {fid_table_path}")


# ============================================================
# CELL 13: LPIPS ANCHOR RETENTION
# ============================================================
print("=" * 70)
print("TABLE 5: LPIPS ANCHOR RETENTION")
print("=" * 70)
!python3 eval/eval_lpips.py \
    --lora_root train/outputs/unlearn/SD-train \
    --output eval/outputs/lpips/lpips_results.json \
    --duo_beta 500 \
    --anchor_beta 500 \
    --concepts "Nudity,Blood,Gun,Horror,Suffer" \
    --per_prompt 4 \
    --device "cuda:0" 2>&1

lpips_path = pathlib.Path("eval/outputs/lpips/lpips_results.json")
if lpips_path.exists():
    print("\n=== LPIPS Results ===")
    print(f"{'Concept':<12s}  {'DUO':>10s}  {'DUO_Anchor':>12s}  {'delta':>10s}")
    print("-" * 48)
    with open(lpips_path) as f:
        data = json.load(f)
    for concept in ("Nudity", "Blood", "Gun", "Horror", "Suffer"):
        if concept in data:
            d = data[concept]
            print(f"{concept:<12s}  {d['lpips_duo']:>10.4f}  {d['lpips_duo_anchor']:>12.4f}  {d['delta']:>+10.4f}")


# ============================================================
# CELL 14: PAPER RESULTS TABLE (Tổng hợp tất cả metrics)
# ============================================================
import json, pathlib

print("=" * 80)
print("FINAL PAPER TABLE: DUO vs DUO-Anchor trên tất cả metrics")
print("=" * 80)
print()

# Collect all results
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}

all_metrics = {}

# 1. Concept Inversion DSR
ci_root = pathlib.Path("eval/outputs/concept_inversion")
for summary_path in ci_root.rglob("concept_inversion_summary.json"):
    all_metrics["concept_inversion"] = load_json(summary_path)

# 2. Ring-A-Bell DSR
rab_root = pathlib.Path("eval/outputs/ring_a_bell")
for summary_path in rab_root.rglob("ring_a_bell_summary.json"):
    all_metrics["ring_a_bell"] = load_json(summary_path)

# 3. FID/CLIP
fid_root = pathlib.Path("eval/outputs/fid_clip")
fid_clip_results = {}
for json_path in fid_root.rglob("fid_clip_*.json"):
    data = load_json(json_path)
    label = json_path.stem.replace("fid_clip_", "")
    fid_clip_results[label] = data
all_metrics["fid_clip"] = fid_clip_results

# 4. LPIPS
lpips_path = pathlib.Path("eval/outputs/lpips/lpips_results.json")
all_metrics["lpips"] = load_json(lpips_path)

# Print DSR tables side by side
attack_types = ["concept_inversion", "ring_a_bell"]
attack_labels = ["Concept Inversion", "Ring-A-Bell"]
concepts = ["Nudity", "Blood", "Gun", "Horror", "Suffer"]

for attack_type, attack_label in zip(attack_types, attack_labels):
    data = all_metrics.get(attack_type, {})
    if not data:
        continue

    print(f"\n  Table: DSR under {attack_label} attack (higher = better)")
    print(f"  {'Concept':<10s}  {'DUO':>8s}  {'DUO-Anchor':>12s}  {'Δ':>8s}")
    print(f"  {'-'*42}")
    for c in concepts:
        row = data.get(c, {})
        duo = row.get("duo")
        anc = row.get("duo-anchor")
        duo_s = f"{duo:.3f}" if duo is not None else "N/A"
        anc_s = f"{anc:.3f}" if anc is not None else "N/A"
        if duo is not None and anc is not None:
            delta_s = f"{anc - duo:+.3f}"
        else:
            delta_s = "N/A"
        print(f"  {c:<10s}  {duo_s:>8s}  {anc_s:>12s}  {delta_s:>8s}")

# Print FID/CLIP
if fid_clip_results:
    print(f"\n  Table: FID / CLIP Score on MS COCO 10k")
    print(f"  {'Model':<20s}  {'CLIP Score':>12s}  {'FID':>10s}")
    print(f"  {'-'*46}")
    for label, data in sorted(fid_clip_results.items()):
        cs = data.get("clip_score_unlearn", "N/A")
        f = data.get("fid", "N/A")
        cs_s = f"{cs:.4f}" if isinstance(cs, (int, float)) else str(cs)
        f_s = f"{f:.4f}" if isinstance(f, (int, float)) else str(f)
        print(f"  {label:<20s}  {cs_s:>12s}  {f_s:>10s}")

# Print LPIPS
if all_metrics.get("lpips"):
    data = all_metrics["lpips"]
    print(f"\n  Table: LPIPS Anchor Retention (lower = better preservation)")
    print(f"  {'Concept':<10s}  {'DUO':>10s}  {'DUO-Anchor':>12s}  {'Δ':>10s}")
    print(f"  {'-'*46}")
    for c in concepts:
        row = data.get(c, {})
        duo = row.get("lpips_duo", "N/A")
        anc = row.get("lpips_duo_anchor", "N/A")
        delta = row.get("delta", "N/A")
        duo_s = f"{duo:.4f}" if isinstance(duo, (int, float)) else str(duo)
        anc_s = f"{anc:.4f}" if isinstance(anc, (int, float)) else str(anc)
        delta_s = f"{delta:+.4f}" if isinstance(delta, (int, float)) else str(delta)
        print(f"  {c:<10s}  {duo_s:>10s}  {anc_s:>12s}  {delta_s:>10s}")

# Save combined results
final_paper = {
    "concept_inversion_dsr": all_metrics.get("concept_inversion", {}),
    "ring_a_bell_dsr": all_metrics.get("ring_a_bell", {}),
    "fid_clip": fid_clip_results,
    "lpips": all_metrics.get("lpips", {}),
}
final_path = "eval/outputs/PAPER_RESULTS_ALL.json"
with open(final_path, "w") as f:
    json.dump(final_paper, f, indent=2)
print(f"\nAll results saved to {final_path}")


# ============================================================
# CELL 15: PACK ALL OUTPUTS FOR DOWNLOAD
# ============================================================
# Output gồm:
#   - train/outputs/  (10 LoRA checkpoints)
#   - eval/outputs/   (tất cả eval results)
#   - datasets/SD/config_auto_anchor.json
print("Packing results for download ...")

# 1. Zip eval outputs (chính — nhỏ ~50 MB)
!cd /kaggle/working && tar -czf DUO-Anchor-eval-results.tar.gz \
    DUO-Anchor/eval/outputs/ 2>/dev/null
!echo "Eval results: /kaggle/working/DUO-Anchor-eval-results.tar.gz"

# 2. Zip model weights (có thể lớn ~2-3 GB)
!cd /kaggle/working && tar -czf DUO-Anchor-model-weights.tar.gz \
    DUO-Anchor/train/outputs/ 2>/dev/null
!echo "Model weights: /kaggle/working/DUO-Anchor-model-weights.tar.gz"

# 3. Zip COCO images (nếu có, ~500 MB)
if os.path.isdir("eval/outputs/fid_clip/images"):
    !cd /kaggle/working && tar -czf DUO-Anchor-coco-images.tar.gz \
        DUO-Anchor/eval/outputs/fid_clip/images/ 2>/dev/null
    !echo "COCO images: /kaggle/working/DUO-Anchor-coco-images.tar.gz"

# Show file sizes
!echo ""
!echo "=== File sizes ==="
!ls -lh /kaggle/working/DUO-Anchor-*.tar.gz 2>/dev/null

from IPython.display import FileLink
print("\nDownload links:")
for fname in ["DUO-Anchor-eval-results.tar.gz", "DUO-Anchor-model-weights.tar.gz",
              "DUO-Anchor-coco-images.tar.gz", "DUO-Anchor-datasets.tar.gz"]:
    fpath = f"/kaggle/working/{fname}"
    if os.path.exists(fpath):
        display(FileLink(fpath))


# ============================================================
# CELL 16: TIME SUMMARY
# ============================================================
import time
print("=" * 60)
print("RUN COMPLETE!")
print("=" * 60)
print("Các file đã tạo:")
print("  - eval/outputs/PAPER_RESULTS_ALL.json  (tổng hợp tất cả metrics)")
print("  - eval/outputs/concept_inversion/      (Concept Inversion attack)")
print("  - eval/outputs/ring_a_bell/            (Ring-A-Bell attack)")
print("  - eval/outputs/fid_clip/               (FID/CLIP on COCO 10k) [MỚI]")
print("  - eval/outputs/lpips/                  (LPIPS anchor retention)")
print("")
print("Download từ FileLink ở Cell 15.")
