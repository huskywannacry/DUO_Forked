export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

# Ring-A-Bell black-box attack on EVERYTHING (nudity + 4 violence subs)
# for BOTH sources. Walltime on Kaggle P100 (1 GPU):
#   * 5 LoRAs/source * 2 sources = 10 GA runs
#   * Each: ~3 min GA + ~1 min gen + ~10 s score = ~5 min
#   * Total: ~50 min  (very cheap, well within Kaggle budget)

base_dir=$(pwd)
save_dir="./outputs"
exp_root="$save_dir/unlearn/SD-train"

cd $base_dir/eval

# ---- Nudity (beta=500) ----
beta=500
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

# ---- Violence sub-LoRAs (beta=1000) ----
beta=1000
python eval_attack_ring_a_bell.py \
    --pretrained "CompVis/stable-diffusion-v1-4" \
    --duo_root    "$exp_root/dpo/$beta" \
    --anchor_root "$exp_root/duo-anchor/$beta" \
    --source both \
    --mode violence_sub \
    --num_attack_prompts 50 \
    --ga_population 100 \
    --ga_generations 50 \
    --output_dir "$base_dir/eval/outputs/ring_a_bell"