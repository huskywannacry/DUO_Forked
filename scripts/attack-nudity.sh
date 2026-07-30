export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
port=50000

base_dir=$(pwd)
save_dir="./outputs"
beta=500                # nudity uses beta=500
exp_root="$save_dir/unlearn/SD-train"

# Attack the Nudity LoRA of DUO_Anchor (our model).
model_root="$exp_root/duo-anchor/$beta"

cd $base_dir/eval

python eval_attack_concept_inversion.py \
    --pretrained "CompVis/stable-diffusion-v1-4" \
    --duo_root "$exp_root/dpo/$beta" \
    --anchor_root "$model_root" \
    --source duo-anchor \
    --mode nudity \
    --num_train_images 4 \
    --ti_steps 3000 \
    --ti_lr 5e-3 \
    --num_eval_prompts 50 \
    --output_dir "$base_dir/eval/outputs/concept_inversion"