export PATH="$HOME/.local/bin:$PATH"

base_dir=$(pwd)

cd $base_dir/datasets/SD
python3 generate_datasets.py \
    --save_dir  $base_dir/datasets/SD   \
    --device    "cuda:0"                \
