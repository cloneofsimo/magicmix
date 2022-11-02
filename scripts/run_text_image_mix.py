import argparse
import os

import torch
from dotenv import load_dotenv
from PIL import Image

from magic_mix import magic_mix_single_image

torch.set_grad_enabled(False)

if __name__ == "__main__":
    load_dotenv(verbose=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, default="input_image.jpg")
    parser.add_argument("--output_dir", type=str, default="contents")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--k_min_ratio", type=float, default=0.3)
    parser.add_argument("--k_max_ratio", type=float, default=0.6)
    parser.add_argument("--nu", type=float, default=0.8)
    parser.add_argument("--guidance_scale_at_mix", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--content_semantics_prompts",
        type=str,
        nargs="+",
        default=["coffee machine", "tiger"],
    )

    args = parser.parse_args()

    image = Image.open(args.input_image).convert("RGB")

    mixed_sementics = magic_mix_single_image(
        layout_image=image,
        num_inference_steps=args.num_inference_steps,
        content_semantics_prompts=args.content_semantics_prompts,
        k_min=int(args.k_min_ratio * args.num_inference_steps),
        k_max=int(args.k_max_ratio * args.num_inference_steps),
        nu=args.nu,
        guidance_scale_at_mix=args.guidance_scale_at_mix,
        seed=args.seed,
        device=args.device,
    )

    for i, pil_image in enumerate(mixed_sementics):
        pil_image.save(
            os.path.join(
                args.output_dir,
                f"image_text_mix_csp_{args.content_semantics_prompts[i]}_mixed_nu:{args.nu}_{i}.png",
            )
        )
