# Implementation of MagicMix with Stable Diffusion

Implementation of MagicMix with Stable Diffusion (https://arxiv.org/abs/2210.16056) in PyTorch. Unofficial version.

## Installations

```bash
pip install git+https://github.com/cloneofsimo/magicmix.git
```

To get it to work in GPU, Install nessary pytorch versions and cuda versions.

## Basic Usage

In `magic_mix`, you can find the implementation of MagicMix with Stable Diffusion.

```python
from magic_mix import magic_mix_single_image

image = Image.open(args.input_image).convert("RGB")

mixed_sementics = magic_mix_single_image(
    layout_image=image,
    num_inference_steps=50,
    content_semantics_prompts=["coffee machine", "tiger"],
    k_min=int(20),
    k_max=int(30),
    nu=0.5,
    guidance_scale_at_mix=7.5,
    seed=args.seed,
    device="cuda:0"
) # mixed sementics is PIL image files...

```

Or simply run the following command to generate mixed images.

```bash
python scripts/run_text_image_mix.py \
    --input_image ./examples/inputs/1.jpg \
    --output_dir ./examples/outputs \
    --num_inference_steps 50 \
    --content_semantics_prompts "coffee machine" "tiger" \
    --k_min 20 \
    --k_max 30 \
    --nu 0.5 \
    --guidance_scale_at_mix 7.5 \
    --seed 0
```
