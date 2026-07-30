export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

# Concept Inversion attack on the Nudity LoRA, for BOTH sources.
# This is the head-to-head comparison table:
#   source=duo         -> paper baseline DUO
#   source=duo-anchor  -> our improvement DUO_Anchor
#
# Output: eval/outputs/concept_inversion/compare_both_beta<N>/concept_inversion_summary.json
#   { "Nudity": { "duo": <dsr>, "duo-anchor": <dsr> } }

base_dir=$(pwd)
save_dir="./outputs"
beta=500                # nudity uses beta=500
exp_root="$save_dir/unlearn/SD-train"

cd $base_dir/eval

python eval_attack_concept_inversion.py \
    --pretrained "CompVis/stable-diffusion-v1-4" \
    --duo_root "$exp_root/dpo/$beta" \
    --anchor_root "$exp_root/duo-anchor/$beta" \
    --source both \
    --mode nudity \
    --num_train_images 4 \
    --ti_steps 3000 \
    --ti_lr 5e-3 \
    --num_eval_prompts 50 \
    --output_dir "$base_dir/eval/outputs/concept_inversion"