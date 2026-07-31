# DUO-Anchor v2: End-to-end Kaggle notebook với Multi-Concept Anchor Auto-Discovery
#
# CẢI TIẾN SO VỚI DUO GỐC:
#   Thay vì hand-curate 8 anchor prompts/concept, dùng CLIP tự động:
#     1. discover_anchors.py — tìm top-20 prompts gần target unsafe trong CLIP text space
#     2. generate_anchors_enhanced.py — sinh ảnh + CLIP image-sim filter + NudeNet filter
#     3. select_hard_anchors — chọn 24 ảnh/concept có similarity 0.4-0.9 (hard anchors)
#   => Giúp L_retain giữ đúng "tương cà" mà xóa "máu me" (selectivity)
#
# Pipeline gồm:
#   1. Clone repo + cài deps
#   2. Prepare data + ENHANCED anchor images (auto-discover, CLIP-filtered)
#   3. Train DUO baseline (5 LoRA, dùng config.json gốc)
#   4. Train DUO_Anchor (5 LoRA, dùng config_auto_anchor.json mới)
#   5. Concept Inversion attack (2 sources × 5 LoRAs) -> bảng DSR
#   6. Ring-A-Bell attack (GA-based) -> bảng DSR
#   7. So sánh LPIPS anchor retention giữa baseline vs enhanced
#
# Walltime ước tính (Kaggle P100, 1 GPU):
#   Setup + data               ~20 min
#   Enhanced anchor discovery  ~15 min (CLIP + SD gen + filter)
#   Train DUO baseline         ~75 min
#   Train DUO_Anchor           ~98 min
#   Concept Inversion attack   ~2.5 h
#   Ring-A-Bell attack         ~50 min
#   TOTAL                      ~6.8 h  (vừa với 12 h background session)
#
# KAGGLE SETTINGS
#   * Accelerator:  GPU P100 (hoặc T4 x2)
#   * Internet:     ON
#   * Background:   ON (Save Version -> Run All -> "Save as background")
#   * Optional:     Add-ons > Secrets > add OPENAI_API_KEY (cho GPT-4o judge
#                   của violence; nếu thiếu sẽ fallback sang CLIP proxy local)

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

!pip install -q -r requirements.txt
!pip install -q --upgrade diffusers transformers accelerate
# Uninstall torchao (0.10.0) which conflicts with peft>=0.17.0 required by new diffusers
!pip uninstall -y torchao
!pip install -q --upgrade 'peft>=0.17.0'
!pip install -q git+https://github.com/notAI-tech/NudeNet.git

import torch
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, "
      f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# ============================================================
# CELL 2: OPTIONAL - load OPENAI_API_KEY từ Kaggle Secrets
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
# CELL 3a: PREPARE DATA + BASIC ANCHOR IMAGES (~25 phút)
# ============================================================
# Generate paired unsafe/safe datasets bằng SDEdit (vẫn dùng config.json gốc)
!bash scripts/prepare-dataset.sh

# Generate basic anchor images (config.json hand-curated anchors, fallback)
!bash scripts/prepare-anchor.sh

# Verify basic anchors
!echo "=== datasets/SD ==="
!ls datasets/SD/
!echo "=== basic anchor images per concept ==="
import subprocess
for c in ["Nudity", "Blood", "Gun", "Horror", "Suffer"]:
    result = subprocess.run(
        f"ls datasets/SD/{c}/anchor/ 2>/dev/null | wc -l",
        shell=True, capture_output=True, text=True
    )
    count = result.stdout.strip()
    print(f"  {c:<8s}  {count if count else '0'} basic anchor images")


# ============================================================
# CELL 3b (~15 min, ~3 GB VRAM): ENHANCED ANCHOR AUTO-DISCOVERY
# ============================================================
# Bước này chạy 2 script:
#   1. discover_anchors.py — dùng CLIP tìm top-20 anchor prompts/concept
#      (không cần GPU vì built-in word pool)
#   2. generate_anchors_enhanced.py — sinh ảnh + CLIP similarity filter
#      + NudeNet safety filter -> chọn hard anchors
#
# Output: config_auto_anchor.json (dùng cho training DUO_Anchor)
#         datasets/SD/<concept>/anchor/ (đã filter, ~24 ảnh/concept)

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

# Verify enhanced anchors
!echo "=== enhanced anchor images per concept ==="
for c in ["Nudity", "Blood", "Gun", "Horror", "Suffer"]:
    result = subprocess.run(
        f"ls datasets/SD/{c}/anchor/ 2>/dev/null | grep -c '\\.jpg'",
        shell=True, capture_output=True, text=True
    )
    count = result.stdout.strip()
    print(f"  {c:<8s}  {count if count else '0'} enhanced anchor images")

!echo "=== config_auto_anchor.json anchor prompts ==="
import subprocess
result = subprocess.run(
    ["python3", "-c", """
import json
with open('datasets/SD/config_auto_anchor.json') as f:
    cfg = json.load(f)
for c in cfg:
    if 'anchor_prompts' in cfg[c]:
        prompts = cfg[c]['anchor_prompts']
        print(f'{c:<10s}: {len(prompts)} prompts')
        for p in prompts[:5]:
            print(f'           - {p}')
        if len(prompts) > 5:
            print(f'           ... and {len(prompts)-5} more')
"""],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)


# ============================================================
# CELL 4: TRAIN DUO BASELINE (~75 phút)
# ============================================================
# Nudity (1 LoRA, beta=500, ~25 min)
!bash scripts/sd-nudity.sh

# Violence (4 sub-LoRAs, beta=1000, ~50 min)
!bash scripts/sd-violence.sh


# ============================================================
# CELL 5: TRAIN DUO + ANCHOR (~98 phút, dùng ENHANCED anchors)
# ============================================================
# Lưu ý: script anchor dùng config_auto_anchor.json (có nhiều anchor hơn,
# đã filter bằng CLIP + NudeNet) thay vì config.json gốc.
#
# Nếu script sd-nudity-anchor.sh / sd-violence-anchor.sh mặc định dùng
# config.json, hãy sửa chúng thành config_auto_anchor.json trước khi chạy.

import subprocess

# Tạo bản sao script anchor dùng config_auto_anchor.json
subprocess.run("cp scripts/sd-nudity-anchor.sh scripts/sd-nudity-anchor-enhanced.sh", shell=True, check=True)
subprocess.run("cp scripts/sd-violence-anchor.sh scripts/sd-violence-anchor-enhanced.sh", shell=True, check=True)

# Sửa config_dir trong script copy
subprocess.run("sed -i 's|config\\.json|config_auto_anchor.json|g' scripts/sd-nudity-anchor-enhanced.sh", shell=True, check=True)
subprocess.run("sed -i 's|config\\.json|config_auto_anchor.json|g' scripts/sd-violence-anchor-enhanced.sh", shell=True, check=True)

# Train
subprocess.run("bash scripts/sd-nudity-anchor-enhanced.sh", shell=True, check=True)
subprocess.run("bash scripts/sd-violence-anchor-enhanced.sh", shell=True, check=True)


# ============================================================
# CELL 6: VERIFY CHECKPOINTS
# ============================================================
import os, pathlib
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
# CELL 7 (LONG, ~2.5h, resumable): CONCEPT INVERSION ATTACK
# ============================================================
# Attack BOTH sources (DUO + DUO_Anchor) trên Nudity + 4 Violence sub-LoRAs.
# Kết quả flush sau MỖI (source, target) pair -> Kaggle timeout cũng
# không mất data, chỉ cần chạy lại cell này để resume.
#
# Tiến trình in ra: [attack:duo/Nudity] DSR = 0.42 (29/50 unsafe)
# Log csv: eval/outputs/concept_inversion/compare_both_beta<N>/dsr_log.csv

!bash scripts/attack-both-all.sh 2>&1 | tee /kaggle/working/attack-both-all.log


# ============================================================
# CELL 8: AGGREGATE FINAL DSR TABLE
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

print("\n" + "=" * 60)
print("FINAL DSR AFTER CONCEPT INVERSION ATTACK")
print("(higher DSR = defense more robust to white-box attack)")
print("=" * 60)
print(f"{'Concept':<12s}  {'DUO':>8s}  {'DUO_Anchor':>12s}  {'delta':>8s}")
print("-" * 48)
for tgt in ("Nudity", "Blood", "Gun", "Horror", "Suffer"):
    row = merged.get(tgt, {})
    duo = row.get("duo", float("nan"))
    anc = row.get("duo-anchor", float("nan"))
    delta = anc - duo if duo == duo and anc == anc else float("nan")
    print(f"{tgt:<12s}  {duo:>8.3f}  {anc:>12.3f}  {delta:>+8.3f}")

with open(out_root / "FINAL_DSR_TABLE.json", "w") as f:
    json.dump(merged, f, indent=2)
print(f"\nSaved to {out_root/'FINAL_DSR_TABLE.json'}")


# ============================================================
# CELL 9: DSR LOGS (csv per beta)
# ============================================================
import pathlib
for log in sorted(out_root.rglob("dsr_log.csv")):
    print(f"\n--- {log.relative_to(pathlib.Path('.'))} ---")
    print(log.read_text())


# ============================================================
# CELL 10 (~50 min): RING-A-BELL ATTACK (BLACK-BOX, GA-BASED)
# ============================================================
# Ring-A-Bell (Tsai et al. 2023): dùng Genetic Algorithm tìm prompt gần
# target unsafe prompt nhất trong CLIP text-embedding space.
#
# Không cần train textual inversion, chạy rất nhanh:
#   * GA ~3 min/LoRA
#   * Generate + score ~2 min/LoRA
#   * Total 10 LoRAs = ~50 min
#
# Script gen kết quả trong eval/outputs/ring_a_bell/compare_both_rab/
#  - ring_a_bell_results.json
#  - ring_a_bell_summary.json
#  - dsr_log.csv

print("=" * 60)
print("RING-A-BELL: Nudity + Violence sub-LoRAs (cả 2 sources)")
print("=" * 60)
!bash scripts/attack-ring-a-bell-all.sh 2>&1 | tee /kaggle/working/attack-ring-a-bell-all.log

print("Ring-A-Bell attack done!")


# ============================================================
# CELL 11: AGGREGATE RING-A-BELL RESULTS
# ============================================================
import json, pathlib
rab_root = pathlib.Path("eval/outputs/ring_a_bell")

rab_merged = {}
for summary_path in rab_root.rglob("ring_a_bell_summary.json"):
    print(f"loading {summary_path}")
    with open(summary_path) as f:
        s = json.load(f)
    for tgt, by_src in s.items():
        rab_merged.setdefault(tgt, {}).update(by_src)

if rab_merged:
    print("\n" + "=" * 60)
    print("RING-A-BELL FINAL DSR TABLE")
    print("(higher DSR = defense more robust to black-box GA attack)")
    print("=" * 60)
    print(f"{'Concept':<12s}  {'DUO':>8s}  {'DUO_Anchor':>12s}  {'delta':>8s}")
    print("-" * 48)
    for tgt in ("Nudity", "Blood", "Gun", "Horror", "Suffer"):
        row = rab_merged.get(tgt, {})
        duo = row.get("duo", float("nan"))
        anc = row.get("duo-anchor", float("nan"))
        delta = anc - duo if duo == duo and anc == anc else float("nan")
        print(f"{tgt:<12s}  {duo:>8.3f}  {anc:>12.3f}  {delta:>+8.3f}")

    with open(rab_root / "FINAL_RING_A_BELL_TABLE.json", "w") as f:
        json.dump(rab_merged, f, indent=2)
    print(f"\nSaved to {rab_root/'FINAL_RING_A_BELL_TABLE.json'}")
else:
    print("\nNo Ring-A-Bell summary files found.")

# Show raw CSV logs
for log in sorted(rab_root.rglob("dsr_log.csv")):
    print(f"\n--- {log} ---")
    print(log.read_text())


# ============================================================
# CELL 12: PACK RESULTS FOR DOWNLOAD
# ============================================================
!cd /kaggle/working && tar -czf DUO-Anchor-results.tar.gz \
    DUO-Anchor/eval/outputs/ 2>/dev/null
from IPython.display import FileLink
FileLink('/kaggle/working/DUO-Anchor-results.tar.gz')