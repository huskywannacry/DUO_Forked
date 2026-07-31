export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

# FID/CLIP evaluation on MS COCO 30k for BOTH sources
# This script generates 30k images per source + reference SD1.4 ~ takes 3h on Kaggle P100.
# Results saved to eval/outputs/fid_clip/

base_dir=$(pwd)
save_dir="./train/outputs"
exp_root="$save_dir/unlearn/SD-train"
output_dir="$base_dir/eval/outputs/fid_clip"
mkdir -p "$output_dir"

echo "============================================================"
echo "FID/CLIP on COCO 30k — Reference SD1.4"
echo "============================================================"

# Generate reference images (SD1.4 prior) + score them
python eval/eval_fid_clip_coco.py \
    --model_root "$exp_root/dpo/500" \
    --output "$output_dir/fid_clip_dpo_500.json" \
    --coco_subset 30000 \
    --gen_batch_size 8 \
    --seed 42 \
    --output_images_dir "$output_dir/images" 2>&1 | tee "$output_dir/fid_clip_ref.log"

echo ""
echo "============================================================"
echo "FID/CLIP — DUO baseline (dpo, beta=500, Nudity)"
echo "============================================================"

# FID/CLIP for DUO (uses Nudity LoRA as representative, same as paper)
python eval/eval_fid_clip_coco.py \
    --model_root "$exp_root/dpo/500" \
    --output "$output_dir/fid_clip_dpo_500.json" \
    --coco_subset 30000 \
    --gen_batch_size 8 \
    --seed 42 \
    --output_images_dir "$output_dir/images" \
    --eval_only 2>&1 | tee -a "$output_dir/fid_clip_dpo_500.log"

echo ""
echo "============================================================"
echo "FID/CLIP — DUO-Anchor (duo-anchor, beta=500, Nudity)"
echo "============================================================"

python eval/eval_fid_clip_coco.py \
    --model_root "$exp_root/duo-anchor/500" \
    --output "$output_dir/fid_clip_anchor_500.json" \
    --coco_subset 30000 \
    --gen_batch_size 8 \
    --seed 42 \
    --output_images_dir "$output_dir/images" \
    --eval_only 2>&1 | tee -a "$output_dir/fid_clip_anchor_500.log"

echo ""
echo "FID/CLIP evaluation complete!"
echo "Results in $output_dir/"
