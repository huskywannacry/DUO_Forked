export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

# Concept Inversion attack on each of the 4 Violence sub-LoRAs, for BOTH
# sources. Output JSON pivots on target concept:
#   { "Blood": {"duo": <dsr>, "duo-anchor": <dsr>},
#     "Gun":   {"duo": <dsr>, "duo-anchor": <dsr>},
#     "Horror":{...},
#     "Suffer":{...} }

base_dir=$(pwd)
save_dir="./outputs"
beta=1000               # violence uses beta=1000
exp_root="$save_dir/unlearn/SD-train"

cd $base_dir/eval

python eval_attack_concept_inversion.py \
    --pretrained "CompVis/stable-diffusion-v1-4" \
    --duo_root "$exp_root/dpo/$beta" \
    --anchor_root "$exp_root/duo-anchor/$beta" \
    --source both \
    --mode violence_sub \
    --num_train_images 4 \
    --ti_steps 3000 \
    --ti_lr 5e-3 \
    --num_eval_prompts 50 \
    --output_dir "$base_dir/eval/outputs/concept_inversion"