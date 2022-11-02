import argparse
import os
from typing import List, Optional

import numpy as np
import PIL
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)
from dotenv import load_dotenv
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def _mix_sementics(
    layout_latents: List[torch.Tensor],
    scheduler,
    unet,
    vae,
    conditional_embedding: torch.Tensor,
    unconditional_embedding: Optional[torch.Tensor] = None,
    nu: float = 0.5,
    k_min: int = 15,
    k_max: int = 30,
    cfg_scale: float = 8.0,
    num_inference_steps: int = 50,
    device: str = "cuda:0",
):
    offset = scheduler.config.get("steps_offset", 0)
    init_timestep = k_max + offset
    init_timestep = min(init_timestep, num_inference_steps)
    t_start = max(num_inference_steps - init_timestep + offset, 0)

    latents = layout_latents[0]

    _do_cfg = (
        True if (cfg_scale >= 1.0 and unconditional_embedding is not None) else False
    )

    if _do_cfg:
        text_embeddings = torch.cat([unconditional_embedding, conditional_embedding])
    else:
        text_embeddings = conditional_embedding

    timesteps = scheduler.timesteps[t_start:].to(device)

    for i, t in enumerate(tqdm(timesteps)):

        latent_model_input = torch.cat([latents] * 2) if _do_cfg else latents

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        noise_pred = unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

        if _do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (
                noise_pred_text - noise_pred_uncond
            )

        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if i < len(layout_latents):
            latents = latents * nu + (1 - nu) * layout_latents[i]

    del layout_latents

    return latents


def _generate_layout_from_latents(
    scheduler,
    init_latents: torch.Tensor,
    noise: torch.Tensor,
    k_min: int = 15,
    k_max: int = 30,
    num_inference_steps: int = 50,
) -> List[torch.Tensor]:

    offset = scheduler.config.get("steps_offset", 0)
    init_timestep = k_min + offset

    ret_layouts = []
    for i in range(k_max - k_min):
        _step = init_timestep + i
        timesteps = scheduler.timesteps[-_step]
        timesteps = torch.tensor([timesteps] * noise.shape[0], device=noise.device)

        _init_latents = init_latents.clone()

        _init_latents = scheduler.add_noise(_init_latents, noise, timesteps)

        ret_layouts.append(_init_latents)

    return list(reversed(ret_layouts))


def _pil_from_latents(vae, latents):
    _latents = 1 / 0.18215 * latents.clone()
    image = vae.decode(_latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    ret_pil_images = [Image.fromarray(image) for image in images]

    return ret_pil_images


def _load_tools(device: str, scheduler_type):

    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        use_auth_token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.float16,
    )

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        use_auth_token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.float16,
    )

    vae.to(device), unet.to(device), text_encoder.to(device)
    scheduler = scheduler_type(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    return vae, unet, text_encoder, tokenizer, scheduler


@torch.no_grad()
@torch.autocast("cuda")
def magic_mix_from_scratch(
    layout_prompts: List[str],
    content_semantics_prompts: List[str],
    device: str = "cuda:0",
    num_inference_steps: int = 30,
    height: int = 512,
    width: int = 512,
    guidance_scale: float = 7.5,
    scheduler_type=LMSDiscreteScheduler,
    return_unmixed_sampled: bool = True,
    guidance_scale_at_mix: float = 7.5,
    k_min: int = 15,
    k_max: int = 30,
    nu: float = 0.5,
    seed=0,
) -> List[Image.Image]:

    """
    Samples from SD, then mixes semantics of the layout and content prompts.
    """
    vae, unet, text_encoder, tokenizer, scheduler = _load_tools(device, scheduler_type)

    generator = torch.manual_seed(seed)
    batch_size = len(layout_prompts)

    layout_text_input = tokenizer(
        layout_prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    content_semantics_text_input = tokenizer(
        content_semantics_prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    layout_cond_embeddings = text_encoder(layout_text_input.input_ids.to(device))[0]
    content_semantics_cond_embeddings = text_encoder(
        content_semantics_text_input.input_ids.to(device)
    )[0]

    # get conditionals, unconditionals
    max_length = layout_text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, layout_cond_embeddings])

    # Latents
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(device)

    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps):

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    if return_unmixed_sampled:
        ret_pil_images = _pil_from_latents(vae, latents)
    else:
        ret_pil_images = []

    noise = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    ).to(device)

    layout_latents = _generate_layout_from_latents(
        scheduler,
        latents,
        noise,
        k_min=k_min,
        k_max=k_max,
        num_inference_steps=num_inference_steps,
    )

    mixed_semantics = _mix_sementics(
        layout_latents,
        scheduler,
        unet,
        vae,
        conditional_embedding=content_semantics_cond_embeddings,
        unconditional_embedding=uncond_embeddings,
        nu=nu,
        k_min=k_min,
        k_max=k_max,
        cfg_scale=guidance_scale_at_mix,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    return _pil_from_latents(vae, mixed_semantics), ret_pil_images


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


@torch.no_grad()
@torch.autocast("cuda")
def magic_mix_single_image(
    layout_image: Image.Image,
    content_semantics_prompts: List[str],
    device: str = "cuda:0",
    num_inference_steps: int = 30,
    scheduler_type=LMSDiscreteScheduler,
    guidance_scale_at_mix: float = 7.5,
    k_min: int = 15,
    k_max: int = 30,
    nu: float = 0.5,
    seed=0,
):
    generator = torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    vae, unet, text_encoder, tokenizer, scheduler = _load_tools(device, scheduler_type)
    scheduler.set_timesteps(num_inference_steps)

    if isinstance(layout_image, PIL.Image.Image):
        layout_image = preprocess(layout_image).to(device)

    init_latent_dist = vae.encode(layout_image).latent_dist
    init_latents = init_latent_dist.sample(generator=generator)
    init_latents = 0.18215 * init_latents

    batch_size = len(content_semantics_prompts)

    height, width = layout_image.shape[-2:]

    noise = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
    ).to(device)

    layout_latents = _generate_layout_from_latents(
        scheduler,
        init_latents,
        noise,
        k_min=k_min,
        k_max=k_max,
        num_inference_steps=num_inference_steps,
    )

    content_semantics_text_input = tokenizer(
        content_semantics_prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    content_semantics_cond_embeddings = text_encoder(
        content_semantics_text_input.input_ids.to(device)
    )[0]

    # get conditionals, unconditionals
    max_length = content_semantics_text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    mixed_semantics = _mix_sementics(
        layout_latents,
        scheduler,
        unet,
        vae,
        conditional_embedding=content_semantics_cond_embeddings,
        unconditional_embedding=uncond_embeddings,
        nu=nu,
        k_min=k_min,
        k_max=k_max,
        cfg_scale=guidance_scale_at_mix,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    return _pil_from_latents(vae, mixed_semantics)
