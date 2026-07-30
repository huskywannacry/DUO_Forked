# Download i2p benchmark prompts from HuggingFace
# Paper: Schramowski et al. "Safe Latent Diffusion" (2023)
# Dataset card: AIML-TUDA/i2p
#
# Paper DUO protocol:
#   - Nudity: prompts in 'sexual' category
#   - Violence: prompts with toxicity >= 0.95
import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="datasets/i2p")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Downloading i2p from HuggingFace (AIML-TUDA/i2p)...")
    try:
        from datasets import load_dataset
    except ImportError:
        os.system("pip install -q datasets")
        from datasets import load_dataset

    ds = load_dataset("AIML-TUDA/i2p", split="train")
    print(f"Total i2p prompts: {len(ds)}")

    cols = ds.column_names
    print("Columns:", cols)

    sexual_path = os.path.join(args.out_dir, "sexual.jsonl")
    violence_path = os.path.join(args.out_dir, "violence.jsonl")

    n_sexual = 0
    n_violence = 0
    with open(sexual_path, "w") as fs, open(violence_path, "w") as fv:
        for ex in ds:
            prompt = ex.get("prompt") or ex.get("text") or ex.get("caption", "")
            if not prompt:
                continue
            toxicity = float(ex.get("toxicity", 0.0))
            category = (ex.get("category", "") or "").lower()

            if "sexual" in category or ex.get("sexual", False):
                fs.write(json.dumps({"prompt": prompt, "toxicity": toxicity}) + "\n")
                n_sexual += 1

            if toxicity >= 0.95:
                fv.write(json.dumps({"prompt": prompt, "toxicity": toxicity}) + "\n")
                n_violence += 1

    print(f"Saved {n_sexual} sexual prompts to {sexual_path}")
    print(f"Saved {n_violence} violence prompts (toxicity>=0.95) to {violence_path}")


if __name__ == "__main__":
    main()
