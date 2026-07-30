export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

# Ring-A-Bell black-box attack on the Nudity LoRA, for BOTH sources.
# Head-to-head comparison: DUO vs DUO_Anchor.

base_dir=$(pwd)
save_dir="./outputs"
beta=500
exp_root="$save_dir/unlearn/SD-train"

cd $base_dir/eval

python eval_attack_ring_a_bell.py \
    --pretrained "CompVis/stable-diffusion-v1-4" \
    --duo_root    "$exp_root/dpo/$beta" \
    --anchor_root "$exp_root/duo-anchor/$beta" \
    --source both \
    --mode nudity \
    --num_attack_prompts 50 \
    --ga_population 100 \
    --ga_generations 50 \
    --output_dir "$base_dir/eval/outputs/ring_a_bell"