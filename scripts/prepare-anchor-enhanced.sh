export PATH="$HOME/.local/bin:$PATH"

base_dir=$(pwd)

# Step 1: Auto-discover anchor prompts using CLIP text space
echo "=== Step 1: Auto-discover anchor prompts ==="
cd $base_dir/datasets/SD
python3 discover_anchors.py \
    --config       $base_dir/datasets/SD/config.json \
    --output       $base_dir/datasets/SD/config_auto_anchor.json \
    --top_k        20 \
    --sim_min      0.55 \
    --sim_max      0.88 \
    --device       "cuda:0"

# Step 2: Generate enhanced anchor images + filter by CLIP + NudeNet
echo ""
echo "=== Step 2: Generate enhanced anchors ==="
python3 generate_anchors_enhanced.py \
    --config                    $base_dir/datasets/SD/config_auto_anchor.json \
    --data_dir                  $base_dir/datasets/SD \
    --device                    "cuda:0" \
    --per_prompt                6 \
    --max_anchors               24 \
    --sim_min                   0.40 \
    --sim_max                   0.90 \
    --diversity_threshold       0.97 \
    --nudenet_filter

echo ""
echo "=== Done ==="
echo "Config:    $base_dir/datasets/SD/config_auto_anchor.json"
echo "Anchors:   $base_dir/datasets/SD/<concept>/anchor/*.jpg"
