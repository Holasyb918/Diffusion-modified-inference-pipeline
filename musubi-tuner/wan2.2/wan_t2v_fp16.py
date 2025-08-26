# https://github.com/kohya-ss/musubi-tuner

import argparse
from datetime import datetime
import gc
import random
import os
import re
import time
import math
import copy
from types import ModuleType, SimpleNamespace
from typing import Tuple, Optional, List, Union, Any, Dict

import torch
import accelerate
from accelerate import Accelerator
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from tqdm import tqdm

from musubi_tuner.dataset import image_video_dataset
from musubi_tuner.networks import lora_wan
from musubi_tuner.utils.lora_utils import filter_lora_state_dict
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file, load_safetensors
from musubi_tuner.wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
import musubi_tuner.wan as wan
from musubi_tuner.wan.modules.model import WanModel, load_wan_model, detect_wan_sd_dtype
from musubi_tuner.wan.modules.vae import WanVAE
from musubi_tuner.wan.modules.t5 import T5EncoderModel
from musubi_tuner.wan.modules.clip import CLIPModel
from musubi_tuner.modules.scheduling_flow_match_discrete import (
    FlowMatchDiscreteScheduler,
)
from musubi_tuner.wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from musubi_tuner.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

try:
    from lycoris.kohya import create_network_from_weights
except:
    pass

from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.hv_generate_video import (
    get_time_flag,
    save_images_grid,
    save_videos_grid,
    synchronize_device,
)
from musubi_tuner.dataset.image_video_dataset import load_video

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerationSettings:
    def __init__(
        self,
        device: torch.device,
        cfg,
        dit_dtype: torch.dtype,
        dit_weight_dtype: Optional[torch.dtype],
        vae_dtype: torch.dtype,
    ):
        self.device = device
        self.cfg = cfg
        self.dit_dtype = dit_dtype
        self.dit_weight_dtype = dit_weight_dtype  # may be None if fp8_scaled, may be float8 if fp8 not scaled
        self.vae_dtype = vae_dtype


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="Wan 2.1 inference script")

    # WAN arguments
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory (Wan 2.1 official).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.",
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++", "vanilla"],
        help="The solver used to sample.",
    )

    parser.add_argument("--dit", type=str, default=None, help="DiT checkpoint path")
    parser.add_argument(
        "--dit_high_noise",
        type=str,
        default=None,
        help="DiT checkpoint path for high noise (optional)",
    )
    parser.add_argument(
        "--offload_inactive_dit", action="store_true", help="Offload DiT model to CPU"
    )
    parser.add_argument(
        "--lazy_loading", action="store_true", help="Enable lazy loading for DiT models"
    )
    parser.add_argument("--vae", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument(
        "--vae_dtype",
        type=str,
        default=None,
        help="data type for VAE, default is bfloat16",
    )
    parser.add_argument(
        "--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU"
    )
    parser.add_argument(
        "--t5", type=str, default=None, help="text encoder (T5) checkpoint path"
    )
    parser.add_argument(
        "--clip", type=str, default=None, help="text encoder (CLIP) checkpoint path"
    )
    # LoRA
    parser.add_argument(
        "--lora_weight",
        type=str,
        nargs="*",
        required=False,
        default=None,
        help="LoRA weight path",
    )
    parser.add_argument(
        "--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier"
    )
    parser.add_argument(
        "--lora_weight_high_noise",
        type=str,
        nargs="*",
        default=None,
        help="LoRA weight path for high noise",
    )
    parser.add_argument(
        "--lora_multiplier_high_noise",
        type=float,
        nargs="*",
        default=None,
        help="LoRA multiplier for high noise",
    )
    parser.add_argument(
        "--include_patterns",
        type=str,
        nargs="*",
        default=None,
        help="LoRA module include patterns",
    )
    parser.add_argument(
        "--exclude_patterns",
        type=str,
        nargs="*",
        default=None,
        help="LoRA module exclude patterns",
    )
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )

    # inference
    parser.add_argument(
        "--prompt", type=str, default=None, help="prompt for generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative prompt for generation, use default negative prompt if not specified",
    )
    parser.add_argument(
        "--video_size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="video size, height and width",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=None,
        help="video length, Default depends on task",
    )
    parser.add_argument("--fps", type=int, default=16, help="video fps, Default is 16")
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=None,
        help="number of inference steps, default depends on task",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="path to save generated video"
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--cpu_noise",
        action="store_true",
        help="Use CPU to generate noise (compatible with ComfyUI). Default is False.",
    )
    parser.add_argument(
        "--timestep_boundary",
        type=float,
        default=None,
        help="Timestep boundary for guidance (0.0 to 1.0). Default depends on task.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale for classifier free guidance. Default depends on task.",
    )
    parser.add_argument(
        "--guidance_scale_high_noise",
        type=float,
        default=None,
        help="Guidance scale for classifier free guidance in high noise model. Default depends on task.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="path to video for video2video inference",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="path to image for image2video inference",
    )
    parser.add_argument(
        "--end_image_path",
        type=str,
        default=None,
        help="path to end image for image2video inference",
    )
    parser.add_argument(
        "--control_path",
        type=str,
        default=None,
        help="path to control video for inference with controlnet. video file or directory with images",
    )
    parser.add_argument(
        "--one_frame_inference",
        type=str,
        default=None,
        help="one frame inference, default is None, comma separated values from 'no_2x', 'no_4x', 'no_post', 'control_indices' and 'target_index'.",
    )
    parser.add_argument(
        "--control_image_path",
        type=str,
        default=None,
        nargs="*",
        help="path to control (reference) image for one frame inference.",
    )
    parser.add_argument(
        "--control_image_mask_path",
        type=str,
        default=None,
        nargs="*",
        help="path to control (reference) image mask for one frame inference.",
    )
    parser.add_argument(
        "--trim_tail_frames",
        type=int,
        default=0,
        help="trim tail N frames from the video before saving",
    )
    parser.add_argument(
        "--cfg_skip_mode",
        type=str,
        default="none",
        choices=["early", "late", "middle", "early_late", "alternate", "none"],
        help="CFG skip mode. each mode skips different parts of the CFG. "
        " early: initial steps, late: later steps, middle: middle steps, early_late: both early and late, alternate: alternate, none: no skip (default)",
    )
    parser.add_argument(
        "--cfg_apply_ratio",
        type=float,
        default=None,
        help="The ratio of steps to apply CFG (0.0 to 1.0). Default is None (apply all steps).",
    )
    parser.add_argument(
        "--slg_layers",
        type=str,
        default=None,
        help="Skip block (layer) indices for SLG (Skip Layer Guidance), comma separated",
    )
    parser.add_argument(
        "--slg_scale",
        type=float,
        default=3.0,
        help="scale for SLG classifier free guidance. Default is 3.0. Ignored if slg_mode is None or uncond",
    )
    parser.add_argument(
        "--slg_start",
        type=float,
        default=0.0,
        help="start ratio for inference steps for SLG. Default is 0.0.",
    )
    parser.add_argument(
        "--slg_end",
        type=float,
        default=0.3,
        help="end ratio for inference steps for SLG. Default is 0.3.",
    )
    parser.add_argument(
        "--slg_mode",
        type=str,
        default=None,
        choices=["original", "uncond"],
        help="SLG mode. original: same as SD3, uncond: replace uncond pred with SLG pred",
    )

    # Flow Matching
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=None,
        help="Shift factor for flow matching schedulers. Default depends on task.",
    )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument(
        "--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8"
    )
    parser.add_argument(
        "--fp8_fast",
        action="store_true",
        help="Enable fast FP8 arithmetic (RTX 4XXX+), only for fp8_scaled",
    )
    parser.add_argument(
        "--fp8_t5", action="store_true", help="use fp8 for Text Encoder model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to use for inference. If None, use CUDA if available, otherwise use CPU",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "flash2", "flash3", "torch", "sageattn", "xformers", "sdpa"],
        help="attention mode",
    )
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=0,
        help="number of blocks to swap in the model",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="video",
        choices=["video", "images", "latent", "both", "latent_images"],
        help="output type",
    )
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata"
    )
    parser.add_argument(
        "--latent_path",
        type=str,
        nargs="*",
        default=None,
        help="path to latent for decode. no inference",
    )
    parser.add_argument(
        "--lycoris", action="store_true", help="use lycoris for inference"
    )
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--compile_args",
        nargs=4,
        metavar=("BACKEND", "MODE", "DYNAMIC", "FULLGRAPH"),
        default=["inductor", "max-autotune-no-cudagraphs", "False", "False"],
        help="Torch.compile settings",
    )

    # New arguments for batch and interactive modes
    parser.add_argument(
        "--from_file", type=str, default=None, help="Read prompts from a file"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: read prompts from console",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.from_file and args.interactive:
        raise ValueError(
            "Cannot use both --from_file and --interactive at the same time"
        )

    if (
        args.prompt is None
        and not args.from_file
        and not args.interactive
        and args.latent_path is None
    ):
        raise ValueError(
            "Either --prompt, --from_file, --interactive, or --latent_path must be specified"
        )

    assert (args.latent_path is None or len(args.latent_path) == 0) or (
        args.output_type == "images" or args.output_type == "video"
    ), "latent_path is only supported for images or video output"

    return args


class WanGenerate:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.gen_settings = get_generation_settings(args)
        self.device, self.cfg, self.dit_dtype, self.dit_weight_dtype, self.vae_dtype = (
            self.gen_settings.device,
            self.gen_settings.cfg,
            self.gen_settings.dit_dtype,
            self.gen_settings.dit_weight_dtype,
            self.gen_settings.vae_dtype,
        )

        # prepare accelerator
        self.mixed_precision = "bf16" if self.dit_dtype == torch.bfloat16 else "fp16"
        self.accelerator = accelerate.Accelerator(mixed_precision=self.mixed_precision)

        self.config = self.cfg
        print("self.device: ", self.device)

        # t5 先加载到 CPU 上
        self.text_encoder = self.load_t5(loading_device="cpu")
        # vae 先加载到 CPU 上
        self.vae = self.load_vae(loading_device="cpu")

        # low noise 后被使用，加载到 CPU 上
        start_time = time.time()
        self.dit_low_noise_model = self.load_dit_model(
            self.args.dit, self.dit_weight_dtype, "cpu", self.dit_weight_dtype
        )
        end_time = time.time()
        print(f"load dit low noise model to cpu time: {end_time - start_time} seconds")

        start_time = time.time()
        # high noise 先被使用，加载到 CPU 上
        self.dit_high_noise_model = self.load_dit_model(
            self.args.dit_high_noise,
            self.dit_weight_dtype,
            "cpu",
            self.dit_weight_dtype,
        )
        end_time = time.time()
        print(f"load dit high noise model to cpu time: {end_time - start_time} seconds")
        self.models = [self.dit_high_noise_model, self.dit_low_noise_model]
        # self.models = [self.dit_low_noise_model, self.dit_high_noise_model]

        print("low noise model device: ", self.dit_low_noise_model.device)
        print("high noise model device: ", self.dit_high_noise_model.device)
        print("text encoder device: ", self.text_encoder.device)
        print("vae device: ", self.vae.device)
        self.scheduler = setup_scheduler(
            self.args, self.gen_settings.cfg, self.gen_settings.device
        )

    def load_vae(self, loading_device):
        vae = WanVAE(
            vae_path=self.args.vae,
            device=loading_device,
            dtype=self.vae_dtype,
            cache_device=None,
        )
        return vae

    def load_t5(self, loading_device):
        checkpoint_path = (
            None
            if self.args.ckpt_dir is None
            else os.path.join(self.args.ckpt_dir, self.config.t5_checkpoint)
        )
        tokenizer_path = (
            None
            if self.args.ckpt_dir is None
            else os.path.join(self.args.ckpt_dir, self.config.t5_tokenizer)
        )

        text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=loading_device,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            weight_path=self.args.t5,
            fp8=self.args.fp8_t5,
        )

        return text_encoder

    def load_dit_model(
        self,
        dit_path,
        dit_weight_dtype,
        loading_device,
        loading_weight_dtype,
        lora_weights_list=None,
        lora_multipliers=None,
        is_from_image=False,
    ):
        # 仅加载 dit 模型
        model = load_wan_model(
            self.config,
            loading_device,
            dit_path,
            self.args.attn_mode,
            False,
            loading_device,
            loading_weight_dtype,
            self.args.fp8_scaled and not self.args.lycoris,
            lora_weights_list=lora_weights_list,
            lora_multipliers=lora_multipliers,
            use_scaled_mm=self.args.fp8_fast,
        )
        # model = model.to(self.device)
        model.eval().requires_grad_(False)
        clean_memory_on_device(self.device)
        return model

    def save_image(self, image, save_path, image_name):
        sample = image.unsqueeze(0)
        one_frame_inference = (
            sample.shape[2] == 1
        )  # check if one frame inference is used
        save_images_grid(
            sample,
            save_path,
            image_name,
            rescale=True,
            create_subdir=not one_frame_inference,
        )
        return save_path


class WanGenerateImageClass(WanGenerate):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.n_prompt = (
            args.negative_prompt
            if args.negative_prompt
            else self.config.sample_neg_prompt
        )

    def prepare_inputs(self, seed, encoded_context):
        noise, inputs = prepare_t2v_inputs(
            self.args,
            self.cfg,
            self.accelerator,
            self.device,
            self.vae,
            encoded_context,
        )
        return noise, inputs

    def encode_prompt(self, prompt):
        # input("before encode prompt")
        self.text_encoder.model = self.text_encoder.model.to(self.device)
        encoded_context = self.text_encoder([prompt], self.device)
        # input("after encode prompt")
        self.text_encoder.model = self.text_encoder.model.to("cpu")
        return encoded_context

    def generate_single_image(self, prompt, seed):
        # setup scheduler
        self.args.seed = seed
        scheduler, timesteps = setup_scheduler(self.args, self.cfg, self.device)
        num_timesteps = len(timesteps)
        apply_cfg_array = [True] * num_timesteps

        slg_start_step = int(self.args.slg_start * num_timesteps)
        slg_end_step = int(self.args.slg_end * num_timesteps)

        # set random generator
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # prepare inputs
        if isinstance(prompt, str):
            context = self.encode_prompt(prompt)
            context_null = self.encode_prompt(self.n_prompt)
            encoded_context = {}
            encoded_context["context"] = context
            encoded_context["context_null"] = context_null
            noise, inputs = self.prepare_inputs(seed, encoded_context)
        else:
            noise, inputs = self.prepare_inputs(seed, prompt)

        # noise, inputs = prepare_t2v_inputs(self.args, self.cfg, self.accelerator, self.device, self.vae)

        latent = noise
        arg_c, arg_null = inputs

        # self.models[0].to(self.device)
        # return run_sampling(self.models, noise, scheduler, timesteps, self.args, self.gen_settings, inputs, self.device, seed_g, self.accelerator, is_i2v=False)

        # latent_storage_device = device if not use_cpu_offload else "cpu"
        # latent = latent.to(latent_storage_device)
        # dit sampling
        prev_high_noise = self.args.timestep_boundary is not None

        if prev_high_noise:
            model = self.models[0]
            guidance_scale = self.args.guidance_scale_high_noise
        else:
            model = self.models[-1]  # use low noise model for low noise steps
            guidance_scale = self.args.guidance_scale
        logger.info(
            f"Starting sampling (high noise: {prev_high_noise}). Models: {len(self.models)}, timestep boundary: {self.args.timestep_boundary}, flow shift: {self.args.flow_shift}, guidance scale: {self.args.guidance_scale}"
        )
        # import pdb; pdb.set_trace()
        model.to(self.device)

        for i, t in enumerate(tqdm(timesteps)):
            # print('i, t', i, t, t/1000.0, self.args.timestep_boundary, timesteps)
            is_high_noise = (
                (t / 1000.0) >= self.args.timestep_boundary
                if self.args.timestep_boundary is not None
                else False
            )

            if not is_high_noise and prev_high_noise:
                guidance_scale = self.args.guidance_scale
                logger.info(
                    f"Switching to low noise at step {i}, t={t}, guidance_scale={guidance_scale}"
                )

                # del model
                model.to("cpu")
                gc.collect()

                if len(self.models) > 1 and (
                    self.args.offload_inactive_dit or self.args.lazy_loading
                ):
                    if self.args.blocks_to_swap > 0:
                        # prepare block swap for low noise model
                        logger.info("Waiting for 5 seconds to finish block swap")
                        time.sleep(5)

                    if self.args.offload_inactive_dit:
                        logger.info(
                            f"Switching model to CPU/GPU for both low and high noise models"
                        )
                        self.models[0].to("cpu")

                        if self.args.blocks_to_swap > 0:
                            # prepare block swap for low noise model
                            self.models[-1].move_to_device_except_swap_blocks(
                                self.device
                            )
                            self.models[-1].prepare_block_swap_before_forward()

                    else:  # lazy loading
                        pass

                    gc.collect()
                    clean_memory_on_device(self.device)

                model = self.models[-1]  # use low noise model for low noise steps

            prev_high_noise = is_high_noise

            # latent is on CPU if use_cpu_offload is True
            latent_model_input = [latent.to(self.device)]
            timestep = torch.stack([t]).to(self.device)

            with self.accelerator.autocast(), torch.no_grad():
                noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0].to(
                    self.device
                )

                apply_cfg = apply_cfg_array[i]  # apply CFG or not
                if apply_cfg:
                    apply_slg = i >= slg_start_step and i < slg_end_step
                    # print(f"Applying SLG: {apply_slg}, i: {i}, slg_start_step: {slg_start_step}, slg_end_step: {slg_end_step}")
                    if self.args.slg_mode == "original" and apply_slg:
                        noise_pred_uncond = model(
                            latent_model_input, t=timestep, **inputs
                        )[0].to(self.device)

                        # apply guidance
                        # SD3 formula: scaled = neg_out + (pos_out - neg_out) * cond_scale
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )

                        # calculate skip layer out
                        skip_layer_out = model(
                            latent_model_input,
                            t=timestep,
                            skip_block_indices=self.args.slg_layers,
                            **inputs,
                        )[0].to(self.device)

                        # apply skip layer guidance
                        # SD3 formula: scaled = scaled + (pos_out - skip_layer_out) * self.slg
                        noise_pred = noise_pred + self.args.slg_scale * (
                            noise_pred_cond - skip_layer_out
                        )
                    elif self.args.slg_mode == "uncond" and apply_slg:
                        # noise_pred_uncond is skip layer out
                        noise_pred_uncond = model(
                            latent_model_input,
                            t=timestep,
                            skip_block_indices=self.args.slg_layers,
                            **inputs,
                        )[0].to(self.device)

                        # apply guidance
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )

                    else:
                        # normal guidance
                        noise_pred_uncond = model(
                            latent_model_input, t=timestep, **arg_null
                        )[0].to(self.device)

                        # apply guidance
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )
                else:
                    noise_pred = noise_pred_cond

                # step
                latent_input = latent.unsqueeze(0)
                temp_x0 = scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent_input,
                    return_dict=False,
                    generator=seed_g,
                )[0]

                # update latent
                latent = temp_x0.squeeze(0)
        model.to("cpu")
        gc.collect()
        clean_memory_on_device(self.device)
        return latent

    def generate_seeds_images(self, prompt: str, prompt_index: int):
        seeds = [0, 42, 666, 3407]

        # 编码 prompt
        context = self.encode_prompt(prompt)
        context_null = self.encode_prompt(self.n_prompt)
        encoded_context = {}
        encoded_context["context"] = context
        encoded_context["context_null"] = context_null

        for seed in seeds:
            start_time = time.time()
            latents = self.generate_single_image(prompt, seed)
            images = self.decode_latent(latents)
            save_name = f"prompt-{str(prompt_index).zfill(3)}_seed-{str(seed).zfill(5)}"
            # save_path = os.path.join(self.args.original_save_dir, save_name)
            # self.args.save_path = save_path

            # save_images(images, self.args, original_base_name=save_name)
            save_path = self.save_image(images, self.args.save_path, save_name)
            end_time = time.time()
            print(f"generate single image time: {end_time - start_time} seconds")
            print(f"image save path: {save_path}")

    def generate_from_file(self, file_path: str):
        with open(file_path, "r") as f:
            prompts = f.read().splitlines()

        for prompt_index, prompt in enumerate(prompts[:]):
            if prompt.startswith("# "):
                prompts.remove(prompt)
                continue

            if not prompt.strip():
                prompts.remove(prompt)
                continue

        for prompt_index, prompt in enumerate(prompts):
            self.generate_seeds_images(prompt, prompt_index=prompt_index)

    def decode_latent(self, latent):
        latent = latent.to(self.device)
        self.vae.to_device(self.device)
        with (
            torch.autocast(device_type=self.device.type, dtype=self.vae.dtype),
            torch.no_grad(),
        ):
            images = self.vae.decode(latent.unsqueeze(0))[0].to(torch.float32).cpu()
        self.vae.to_device("cpu")
        return images


def apply_overrides(
    args: argparse.Namespace, overrides: Dict[str, Any]
) -> argparse.Namespace:
    """Apply overrides to args

    Args:
        args: Original arguments
        overrides: Dictionary of overrides

    Returns:
        argparse.Namespace: New arguments with overrides applied
    """
    args_copy = copy.deepcopy(args)

    for key, value in overrides.items():
        if key == "video_size_width":
            args_copy.video_size[1] = value
        elif key == "video_size_height":
            args_copy.video_size[0] = value
        else:
            setattr(args_copy, key, value)

    return args_copy


def get_task_defaults(
    task: str, size: Optional[Tuple[int, int]] = None
) -> Tuple[int, float, int, bool]:
    """Return default values for each task

    Args:
        task: task name (t2v, t2i, i2v etc.)
        size: size of the video (width, height)

    Returns:
        Tuple[int, Optional[float], float, float, float, int, bool]: (infer_steps, boundary, flow_shift, guidance_scale, guidance_scale_high_noise, video_length, needs_clip)
    """
    width, height = size if size else (0, 0)

    cfg = WAN_CONFIGS[task]

    infer_steps = cfg.sample_steps
    boundary = cfg.boundary  # may be None
    flow_shift = cfg.sample_shift
    guidance_scale = cfg.sample_guide_scale[0]
    guidance_scale_high_noise = (
        cfg.sample_guide_scale[1] if len(cfg.sample_guide_scale) > 1 else guidance_scale
    )

    video_length = (
        1 if "t2i" in task else 81
    )  # default video length for t2i is 1, for others is 81

    if not cfg.v2_2:
        # Wan2.1
        needs_clip = "i2v" in task
        if "i2v" in task and (
            (width == 832 and height == 480) or (width == 480 and height == 832)
        ):
            #  I2V
            flow_shift = 3.0
    else:
        # Wan2.2
        needs_clip = False

    return (
        infer_steps,
        boundary,
        flow_shift,
        guidance_scale,
        guidance_scale_high_noise,
        video_length,
        needs_clip,
    )


def setup_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate and set default values for optional arguments

    Args:
        args: command line arguments

    Returns:
        argparse.Namespace: updated arguments
    """
    # Get default values for the task
    (
        infer_steps,
        boundary,
        flow_shift,
        guidance_scale,
        guidance_scale_high_noise,
        video_length,
        needs_clip,
    ) = get_task_defaults(args.task, tuple(args.video_size))

    # Apply default values to unset arguments
    if args.infer_steps is None:
        args.infer_steps = infer_steps
    if args.timestep_boundary is None:
        args.timestep_boundary = boundary
    if args.flow_shift is None:
        args.flow_shift = flow_shift
    if args.guidance_scale is None:
        args.guidance_scale = guidance_scale
    if args.guidance_scale_high_noise is None:
        args.guidance_scale_high_noise = guidance_scale_high_noise
    if args.video_length is None:
        args.video_length = video_length

    # Force video_length to 1 for t2i tasks
    if "t2i" in args.task:
        assert args.video_length == 1, f"video_length should be 1 for task {args.task}"
    if args.timestep_boundary is not None:
        if args.timestep_boundary > 1.0:
            logger.warning(
                f"timestep_boundary {args.timestep_boundary} is greater than 1.0, setting to {args.timestep_boundary / 1000.0}"
            )
            args.timestep_boundary = args.timestep_boundary / 1000.0

    # parse slg_layers
    if args.slg_layers is not None:
        args.slg_layers = list(map(int, args.slg_layers.split(",")))

    return args


def check_inputs(args: argparse.Namespace) -> Tuple[int, int, int]:
    """Validate video size and length

    Args:
        args: command line arguments

    Returns:
        Tuple[int, int, int]: (height, width, video_length)
    """
    height = args.video_size[0]
    width = args.video_size[1]
    size = f"{width}*{height}"

    if size not in SUPPORTED_SIZES[args.task]:
        logger.warning(
            f"Size {size} is not supported for task {args.task}. Supported sizes are {SUPPORTED_SIZES[args.task]}."
        )

    video_length = args.video_length

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
        )

    return height, width, video_length


def calculate_dimensions(
    video_size: Tuple[int, int], video_length: int, config
) -> Tuple[Tuple[int, int, int, int], int]:
    """calculate dimensions for the generation

    Args:
        video_size: video frame size (height, width)
        video_length: number of frames in the video
        config: model configuration

    Returns:
        Tuple[Tuple[int, int, int, int], int]:
            ((channels, frames, height, width), seq_len)
    """
    height, width = video_size
    frames = video_length

    # calculate latent space dimensions
    lat_f = (frames - 1) // config.vae_stride[0] + 1
    lat_h = height // config.vae_stride[1]
    lat_w = width // config.vae_stride[2]

    # calculate sequence length
    seq_len = math.ceil(
        (lat_h * lat_w) / (config.patch_size[1] * config.patch_size[2]) * lat_f
    )

    return ((16, lat_f, lat_h, lat_w), seq_len)


def prepare_t2v_inputs(
    args: argparse.Namespace,
    config,
    accelerator: Accelerator,
    device: torch.device,
    vae: Optional[WanVAE] = None,
    encoded_context: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Tuple[dict, dict]]:
    """Prepare inputs for T2V

    Args:
        args: command line arguments
        config: model configuration
        accelerator: Accelerator instance
        device: device to use
        vae: VAE model for control video encoding
        encoded_context: Pre-encoded text context

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
            (noise, context, context_null, (arg_c, arg_null))
    """
    # Prepare inputs for T2V
    # calculate dimensions and sequence length
    height, width = args.video_size
    frames = args.video_length
    (_, lat_f, lat_h, lat_w), seq_len = calculate_dimensions(
        args.video_size, args.video_length, config
    )
    target_shape = (16, lat_f, lat_h, lat_w)

    # configure negative prompt
    n_prompt = (
        args.negative_prompt if args.negative_prompt else config.sample_neg_prompt
    )

    # set seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        # ComfyUI compatible noise
        seed_g = torch.manual_seed(seed)

    if encoded_context is None:
        # load text encoder
        text_encoder = load_text_encoder(args, config, device)
        text_encoder.model.to(device)

        # encode prompt
        with torch.no_grad():
            if args.fp8_t5:
                with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                    context = text_encoder([args.prompt], device)
                    context_null = text_encoder([n_prompt], device)
            else:
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)

        # free text encoder and clean memory
        del text_encoder
        clean_memory_on_device(device)
    else:
        # Use pre-encoded context
        context = encoded_context["context"]
        context_null = encoded_context["context_null"]

    # Fun-Control: encode control video to latent space
    if config.is_fun_control:
        # TODO use same resizing as for image
        logger.info(f"Encoding control video to latent space")
        # C, F, H, W
        control_video = load_control_video(args.control_path, frames, height, width).to(
            device
        )
        vae.to_device(device)
        with torch.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
            control_latent = vae.encode([control_video])[0]
        y = torch.concat(
            [control_latent, torch.zeros_like(control_latent)], dim=0
        )  # add control video latent
        vae.to_device("cpu")
    else:
        y = None

    # generate noise
    noise = torch.randn(
        target_shape,
        dtype=torch.float32,
        generator=seed_g,
        device=device if not args.cpu_noise else "cpu",
    )
    noise = noise.to(device)

    # prepare model input arguments
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}
    if y is not None:
        arg_c["y"] = [y]
        arg_null["y"] = [y]

    return noise, (arg_c, arg_null)


def setup_scheduler(
    args: argparse.Namespace, config, device: torch.device
) -> Tuple[Any, torch.Tensor]:
    """setup scheduler for sampling

    Args:
        args: command line arguments
        config: model configuration
        device: device to use

    Returns:
        Tuple[Any, torch.Tensor]: (scheduler, timesteps)
    """
    if args.sample_solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(args.infer_steps, device=device, shift=args.flow_shift)
        timesteps = scheduler.timesteps
    elif args.sample_solver == "dpm++":
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sampling_sigmas = get_sampling_sigmas(args.infer_steps, args.flow_shift)
        timesteps, _ = retrieve_timesteps(
            scheduler, device=device, sigmas=sampling_sigmas
        )
    elif args.sample_solver == "vanilla":
        scheduler = FlowMatchDiscreteScheduler(
            num_train_timesteps=config.num_train_timesteps, shift=args.flow_shift
        )
        scheduler.set_timesteps(args.infer_steps, device=device)
        timesteps = scheduler.timesteps

        # FlowMatchDiscreteScheduler does not support generator argument in step method
        org_step = scheduler.step

        def step_wrapper(
            model_output: torch.Tensor,
            timestep: Union[int, torch.Tensor],
            sample: torch.Tensor,
            return_dict: bool = True,
            generator=None,
        ):
            return org_step(model_output, timestep, sample, return_dict=return_dict)

        scheduler.step = step_wrapper
    else:
        raise NotImplementedError("Unsupported solver.")

    return scheduler, timesteps


def save_i1mages(
    sample: torch.Tensor,
    args: argparse.Namespace,
    original_base_name: Optional[str] = None,
) -> str:
    """Save images to directory

    Args:
        sample: Video tensor
        args: command line arguments
        original_base_name: Original base name (if latents are loaded from files)

    Returns:
        str: Path to saved images directory
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    image_name = f"{time_flag}_{seed}{original_name}"
    sample = sample.unsqueeze(0)
    one_frame_inference = sample.shape[2] == 1  # check if one frame inference is used
    save_images_grid(
        sample,
        save_path,
        image_name,
        rescale=True,
        create_subdir=not one_frame_inference,
    )
    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    device = torch.device(args.device)

    cfg = WAN_CONFIGS[args.task]

    # select dtype
    dit_dtype = (
        detect_wan_sd_dtype(args.dit) if args.dit is not None else torch.bfloat16
    )
    if dit_dtype.itemsize == 1:
        # if weight is in fp8, use bfloat16 for DiT (input/output)
        dit_dtype = torch.bfloat16
        if args.fp8_scaled:
            raise ValueError(
                "DiT weights is already in fp8 format, cannot scale to fp8. Please use fp16/bf16 weights / DiTの重みはすでにfp8形式です。fp8にスケーリングできません。fp16/bf16の重みを使用してください"
            )

    dit_weight_dtype = dit_dtype  # default
    if args.fp8_scaled:
        dit_weight_dtype = (
            None  # various precision weights, so don't cast to specific dtype
        )
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn

    vae_dtype = (
        str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else dit_dtype
    )
    logger.info(
        f"Using device: {device}, DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}, VAE precision: {vae_dtype}"
    )

    gen_settings = GenerationSettings(
        device=device,
        cfg=cfg,
        dit_dtype=dit_dtype,
        dit_weight_dtype=dit_weight_dtype,
        vae_dtype=vae_dtype,
    )
    return gen_settings


if __name__ == "__main__":
    args = parse_args()
    device = (
        args.device
        if args.device is not None
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    args.device = device
    if not hasattr(args, "original_save_dir"):
        args.original_save_dir = args.save_path

    args = setup_args(args)

    wan_generate_image_class = WanGenerateImageClass(args)
    # input prompt is a txt file path
    wan_generate_image_class.generate_from_file(args.prompt)
    exit(-1)

    # input prompt is a string, test multi seeds
    wan_generate_image_class.generate_seeds_images(args.prompt, prompt_index=0)
    exit(-1)

    # test original
    latent = wan_generate_image_class.generate_single_image(args.prompt, seed=0)
    wan_generate_image_class.models[-1].to("cpu")
    image = wan_generate_image_class.decode_latent(latent)

    save_images(image, args, original_base_name="loop.png")
