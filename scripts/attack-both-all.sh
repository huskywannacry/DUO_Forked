export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

# Concept Inversion attack on EVERYTHING (nudity + 4 violence subs) for BOTH
# sources. Runs as TWO sequential calls because nudity uses beta=500 and
# violence uses beta=1000. Each call writes its own JSON; aggregate at
# the notebook level.
#
# Walltime on Kaggle P100 (1 GPU):
#   * 10 LoRAs x ~13 min per TI run + ~3 min gen/score = ~2.5 h total.
# * Fits comfortably inside the 12 h background-session budget.

base_dir=$(pwd)
save_dir="./train/outputs"
exp_root="$save_dir/unlearn/SD-train"

cd $base_dir/eval

# ---- Nudity (beta=500) ----
beta=500
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

# ---- Violence sub-LoRAs (beta=1000) ----
beta=1000
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