export PATH="$HOME/.local/bin:$PATH"

base_dir=$(pwd)

cd $base_dir/datasets/SD
python3 generate_anchors.py \
    --config_dir $base_dir/datasets/SD/config.json \
    --data_dir   $base_dir/datasets/SD              \
    --device     "cuda:0"                            \
    --per_prompt 4
