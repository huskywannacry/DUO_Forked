# DUO
# Copyright (c) 2024-present NAVER loud Corp.
# Apache-2.0C

import argparse
import gc
import json
import os

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

weight_dtype = torch.float16
device = "cuda"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        required=True,
        help="Input prompt",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unlearn_model_path",
        type=str,
        default="train/outputs/unlearn/SD-train/dpo/1000",
        required=False,
        help="Path to unlearned lora path",
    )
    parser.add_argument(
        "--exp_type",
        type=str,
        default="violence",
        required=False,
        help="'violence' or 'nudity'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


def load_sd_dpo(args):
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    ).to(device)

    if args.exp_type == "violence":
        config_list = ["Blood", "Gun", "Horror", "Suffer"]
        for config_name in config_list:
            lora_path = os.path.join(
                args.unlearn_model_path,
                config_name,
                "pytorch_lora_weights.safetensors",
            )
            if not os.path.exists(lora_path):
                # fallback to checkpoint-500 (legacy)
                lora_path = os.path.join(
                    args.unlearn_model_path,
                    config_name,
                    "checkpoint-500",
                    "pytorch_lora_weights.safetensors",
                )
            pipe.load_lora_weights(lora_path, adapter_name=config_name)
        pipe.set_adapters(config_list, adapter_weights=[1, 1, 1, 1])
    else:
        lora_path = os.path.join(
            args.unlearn_model_path, "Nudity", "pytorch_lora_weights.safetensors"
        )
        if not os.path.exists(lora_path):
            lora_path = os.path.join(
                args.unlearn_model_path, "Nudity",
                "checkpoint-1000", "pytorch_lora_weights.safetensors",
            )
        pipe.load_lora_weights(lora_path)
    return pipe


if __name__ == "__main__":
    args = parse_args()
    pipe = load_sd_dpo(args)

    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device, weight_dtype)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    generator = torch.Generator(device)
    images = pipe(args.prompt, generator=generator.manual_seed(args.seed)).images
    images[0].save(args.output)

    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Saved: {args.output}")
