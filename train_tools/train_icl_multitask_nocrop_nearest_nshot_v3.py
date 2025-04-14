#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, StableDiffusionPipeline #, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available

import sys
sys.path.append('./')

from evaluation_util.data.dataset import FSSDataset

from diffews.models.unet_2d_condition_v2 import MyUNet2DConditionModel as UNet2DConditionModel
from marigold.util.scheduler_customized import DDIMSchedulerCustomized as DDIMScheduler
from marigold.util.scheduler_customized import DDPMSchedulerCustomized as DDPMScheduler

from PIL import Image
import os.path as osp
import psutil


from datasets.arrow_dataset import concatenate_datasets
# from diffusers_custom.my_image import MyImage
from marigold.util.image_util import chw2hwc, colorize_depth_maps

if is_wandb_available():
    import wandb

import time

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.23.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

def save_model_card(
    args,
    repo_id: str,
    images=None,
    repo_folder=None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {args.pretrained_model_name_or_path}
datasets:
- {args.dataset_name}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
inference: true
---
    """
    model_card = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_card += wandb_info

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def pyramid_noise_like(x, timesteps, num_train_timesteps, discount=0.9):
  b, c, origin_w, origin_h = x.shape
  u = nn.Upsample(size=(origin_w, origin_h), mode='bilinear')
  noise = torch.randn_like(x)
  
  for i in range(6):
    r = random.random() * 2 + 2 # Rather than always going 2x, 
    w, h = max(1, int(origin_w/(r**i))), max(1, int(origin_h/(r**i)))
    # import pdb
    # pdb.set_trace()
    discount_i = (discount * timesteps / num_train_timesteps)[...,None,None,None]
    noise += u(torch.randn(b, c, w, h).to(x)) * (discount_i **i)

    if w==1 or h==1: break 
  return noise / noise.std() # Scale back to unit variance




def get_test_dataloader(args):

    test_dataloader_list = []
    dataset_list = []
    if args.mode == 'seg':
        dam = None
    else:
        raise ValueError

    return test_dataloader_list, dam, dataset_list
        


def get_train_dataset(args, accelerator):
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
                num_proc=psutil.cpu_count(),
                # num_proc=1,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    # dataset = concatenate_datasets([dataset['bedlam'], dataset['humman']], split=datasets.Split.TRAIN)
    if "train" in dataset.keys():
        dataset = concatenate_datasets([dataset["train"]], split=datasets.Split.TRAIN)
    else:
        all_datasets = []
        for k in dataset.keys():
            # dataset[k].info.features['image'] = MyImage()
            # dataset[k].info.features['seg_conditioning_image'] = MyImage()
            # dataset[k].info = myinfo
            all_datasets.append(dataset[k])
        
        dataset = concatenate_datasets(all_datasets, split=datasets.Split.TRAIN)

    column_names = dataset.column_names

    # 6. Get the column names for input/target.
    if args.image_ref_column is None:
        image_ref_column = column_names[0]
        logger.info(f"image ref column defaulting to {image_ref_column}")
    else:
        image_ref_column = args.image_ref_column
        if image_ref_column not in column_names:
            raise ValueError(
                f"`--image_ref_column` value '{args.image_ref_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
    
    if args.image_tag_column is None:
        image_tag_column = column_names[1]
        logger.info(f"image tag column defaulting to {image_tag_column}")
    else:
        image_tag_column = args.image_tag_column
        if image_tag_column not in column_names:
            raise ValueError(
                f"`--image_tag_column` value '{args.image_tag_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )


    if args.caption_column is None:
        caption_column = column_names[4]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_ref_column is None:
        conditioning_image_ref_column = column_names[2]
        logger.info(f"conditioning image ref column defaulting to {conditioning_image_ref_column}")
    else:
        conditioning_image_ref_column = args.conditioning_image_ref_column
        if conditioning_image_ref_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_ref_column` value '{args.conditioning_image_ref_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
    
    if args.conditioning_image_tag_column is None:
        conditioning_image_tag_column = column_names[2]
        logger.info(f"conditioning image tag column defaulting to {conditioning_image_tag_column}")
    else:
        conditioning_image_tag_column = args.conditioning_image_tag_column
        if conditioning_image_tag_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_tag_column` value '{args.conditioning_image_tag_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
            
    with accelerator.main_process_first():
        # train_dataset = dataset["train"].shuffle(seed=args.seed)
        train_dataset = dataset.shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))

    return train_dataset


def prepare_train_dataset(args, dataset, accelerator, tokenizer):
    image_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution,args.resolution), interpolation=transforms.InterpolationMode.NEAREST),
            # transforms.CenterCrop(args.resolution),
            # transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), 
        ]
    )
    
    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution,args.resolution), interpolation=transforms.InterpolationMode.NEAREST),
            # transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            # transforms.RandomErasing(p=0.25),
        ]
    )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[args.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{args.caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def preprocess_train(examples):

        def preprocess_image(image_tensor):
            delta_h = image_tensor.shape[1] - args.resolution
            delta_w = image_tensor.shape[2] - args.resolution

            top = delta_h // 2
            left = delta_w // 2

            crop_coords_top_left = torch.tensor([top, left]) 

            image = transforms.functional.crop(
                image_tensor, top=top, left=left, height=args.resolution, width=args.resolution
            )
            return image, crop_coords_top_left


        ref_original_size_list = [torch.tensor(image.size) for image in examples[args.image_ref_column]]
        tag_original_size_list = [torch.tensor(image.size) for image in examples[args.image_tag_column]]
        target_size_list = [torch.tensor([args.resolution, args.resolution]) for i in range(len(examples[args.image_tag_column]))]

        images_ref = [image.convert("RGB") for image in examples[args.image_ref_column]]
        images_ref = [image_transforms(image) for image in images_ref]

        images_tag = [image.convert("RGB") for image in examples[args.image_tag_column]]
        images_tag = [image_transforms(image) for image in images_tag]

        conditioning_images_ref = [image.convert("RGB") for image in examples[args.conditioning_image_ref_column]]
        conditioning_images_ref = [conditioning_image_transforms(image) for image in conditioning_images_ref]

        conditioning_images_tag = [image.convert("RGB") for image in examples[args.conditioning_image_tag_column]]
        conditioning_images_tag = [conditioning_image_transforms(image) for image in conditioning_images_tag]

        # import pdb
        # pdb.set_trace()

        examples["input_ids"] = tokenize_captions(examples)

        crop_coords_top_left_ref_list = []
        crop_coords_top_left_tag_list = []

        for idx, (image_ref, image_tag, conditioning_image_ref, conditioning_image_tag) in enumerate(zip(images_ref, images_tag, conditioning_images_ref, conditioning_images_tag)):

            images_ref[idx] = image_ref
            images_tag[idx] = image_tag
            conditioning_images_ref[idx] = conditioning_image_ref
            conditioning_images_tag[idx] = conditioning_image_tag

        examples["images_ref"] = images_ref
        examples["images_tag"] = images_tag
        examples["conditioning_images_ref"] = conditioning_images_ref
        examples["conditioning_images_tag"] = conditioning_images_tag
        # examples["crop_coords_top_left_ref_list"] = crop_coords_top_left_ref_list
        # examples["crop_coords_top_left_tag_list"] = crop_coords_top_left_tag_list
        examples["ref_original_size_list"] = ref_original_size_list
        examples["tag_original_size_list"] = tag_original_size_list
        examples["target_size"] = target_size_list

        return examples


    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset


def collate_fn(examples):
    imgs_ref = torch.stack([example["images_ref"] for example in examples])
    imgs_ref = imgs_ref.to(memory_format=torch.contiguous_format).float()

    imgs_tag = torch.stack([example["images_tag"] for example in examples])
    imgs_tag = imgs_tag.to(memory_format=torch.contiguous_format).float()
    
    conditioning_images_ref = torch.stack([example["conditioning_images_ref"] for example in examples])
    conditioning_images_ref = conditioning_images_ref.to(memory_format=torch.contiguous_format).float()

    conditioning_images_tag = torch.stack([example["conditioning_images_tag"] for example in examples])
    conditioning_images_tag = conditioning_images_tag.to(memory_format=torch.contiguous_format).float()

    ref_original_size_list = torch.stack([example["ref_original_size_list"] for example in examples])
    tag_original_size_list = torch.stack([example["tag_original_size_list"] for example in examples])
    # crop_coords_top_left_ref_list = torch.stack([example["crop_coords_top_left_ref_list"] for example in examples])
    # crop_coords_top_left_tag_list = torch.stack([example["crop_coords_top_left_tag_list"] for example in examples])
    target_size = torch.stack([example["target_size"] for example in examples])

    input_ids = torch.stack([example["input_ids"] for example in examples])

    res_dict = {
        "imgs_ref": imgs_ref,
        "imgs_tag": imgs_tag,
        "conditioning_images_ref": conditioning_images_ref,
        "conditioning_images_tag": conditioning_images_tag,
        "input_ids": input_ids,
        "ref_original_size_list": ref_original_size_list,
        "tag_original_size_list": tag_original_size_list,
        # "crop_coords_top_left_ref_list": crop_coords_top_left_ref_list,
        # "crop_coords_top_left_tag_list": crop_coords_top_left_tag_list,
        "target_size": target_size,
    }

    if 'prompt_embeds' in examples[0]:
        prompt_ids = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
        add_text_embeds = torch.stack([torch.tensor(example["text_embeds"]) for example in examples])
        add_time_ids = torch.stack([torch.tensor(example["time_ids"]) for example in examples])
        # add_image_embeds = torch.stack([torch.tensor(example["image_embeds"]) for example in examples])
        res_dict["prompt_ids"] = prompt_ids,
        res_dict["unet_added_conditions"] = {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
    else:
        text = [example['text'] for example in examples]
        res_dict['text'] = text

    return res_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['depth', 'normal', 'seg'],
        default="seg",
        help="inference mode.",
    )
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/stable-diffusion-2-1-ref8inchannels-tag4inchannels",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--scheduler_load_path",
        type=str,
        default='./scheduler_1.0_1.0',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--gt_data_root",
        type=str,
        default='/test/xugk/data/data_metricdepth',
        required=False,
        help="gt data root",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="./datasets/hypersim_icl/multitaskv1",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_ref_column", type=str, default="img_ref", help="The column of the dataset containing an image."
    )
    parser.add_argument(
    "--image_tag_column", type=str, default="img_tag", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--conditioning_image_ref_column",
        type=str,
        default="ref_conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--conditioning_image_tag_column",
        type=str,
        default="tag_conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_images` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs/debug",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='./cache',
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=30000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="polynomial",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_scheduler_power", type=float, default=1.0, help="Lr scheduler power."
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd21_train_dis",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    # add train_time_steps
    parser.add_argument(
        "--train_timestep",
        type=int,
        default=1,
        help="timesteps for training",
    )

    parser.add_argument(
        "--nshot",
        type=int,
        default=1,
        help="number of shots for training",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="number of shots for validation",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id



    if args.scheduler_load_path is not None:
        # noise_scheduler = DDPMScheduler.from_pretrained(args.scheduler_load_path, subfolder="scheduler")
        noise_scheduler_ddim = DDIMScheduler.from_pretrained(args.scheduler_load_path, subfolder="scheduler")
    else:
        # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        noise_scheduler_ddim = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")


    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        if args.pretrained_vae_path is not None:
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_vae_path, subfolder="vae", revision=args.revision, variant=args.variant
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
            )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    args.benchmark = 'coco'
    args.datapath = args.train_data_dir
    args.use_original_imgsize = False
    #args.nshot = 2
    args.bsz = args.train_batch_size
    args.nworker = args.dataloader_num_workers

    print ('fold:', args.fold)

    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_train_nshot = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', args.nshot)
   
    eval_results_all = {}
    train_dataloader = dataloader_train_nshot 
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        power=args.lr_scheduler_power,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # for i in range(len(test_dataloader_list)):
    #     test_dataloader_list[i] = accelerator.prepare(test_dataloader_list[i])

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        tracker_config.pop("validation_images")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    #logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Learning Rate = {args.learning_rate}")
    logger.info(f"  Training timesteps = {args.train_timestep}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if os.path.exists(args.resume_from_checkpoint):
                accelerator.load_state(args.resume_from_checkpoint, map_location="cpu")
            else:
                accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    temp_input_ids = torch.load('temp_input_ids.pt').to(accelerator.device)
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch_ in enumerate(dataloader_train_nshot):
            with accelerator.accumulate(unet):
                batch = {}
                batch['imgs_ref'] = batch_['support_imgs'][0].to(accelerator.device)
                batch['imgs_tag'] = batch_['query_img'].to(accelerator.device)
                batch['conditioning_images_ref'] = batch_['support_masks'][0]
                # convert mask 0~1 to -1~1
                batch['conditioning_images_ref'] = batch['conditioning_images_ref'] * 2 - 1
                batch['conditioning_images_tag'] = batch_['query_mask']
                batch['conditioning_images_tag'] = batch['conditioning_images_tag'] * 2 - 1
                # for mask 1x512x512 -> 1x3x512x512
                batch['conditioning_images_ref'] = batch['conditioning_images_ref'].unsqueeze(1).repeat(1,3,1,1).to(accelerator.device)
                batch['conditioning_images_tag'] = batch['conditioning_images_tag'].unsqueeze(1).repeat(1,3,1,1).to(accelerator.device)
                # random select 1~5 shots
                max_nshot = args.nshot
                temp_nshot = random.randint(1, max_nshot)
                indices = random.sample(range(0, max_nshot), temp_nshot)
                batch['imgs_ref'] = batch['imgs_ref'][indices]  
                batch['conditioning_images_ref'] = batch['conditioning_images_ref'][indices]
                batch['input_ids'] = temp_input_ids
                # Convert reference and target images to latent space
                # check mask area 
                # ref_mask_area = (batch["conditioning_images_ref"]>0).sum(dim=(1,2,3)).item()
                # tag_mask_area = (batch["conditioning_images_tag"]>0).sum(dim=(1,2,3)).item()
                # print('ref_mask_area:', ref_mask_area, 'tag_mask_area:', tag_mask_area)
                latents_ref = vae.encode(batch["imgs_ref"].to(weight_dtype)).latent_dist.sample()
                latents_ref = latents_ref * vae.config.scaling_factor # [bs, 4, 96, 96]
                

                latents_tag = vae.encode(batch["imgs_tag"].to(weight_dtype)).latent_dist.sample() # input 1x3x512x512 -1~1
                latents_tag = latents_tag * vae.config.scaling_factor # [bs, 4, 96, 96]

                #  seg gt images
                cond_latents_ref = vae.encode(batch["conditioning_images_ref"].to(weight_dtype)).latent_dist.sample()
                cond_latents_ref = cond_latents_ref * vae.config.scaling_factor 
                cond_latents_tag = vae.encode(batch["conditioning_images_tag"].to(weight_dtype)).latent_dist.sample()
                cond_latents_tag = cond_latents_tag * vae.config.scaling_factor 

                latents_rgb_cond_ref = torch.cat([latents_ref, cond_latents_ref], dim=1)


                bsz = latents_tag.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.tensor([1*args.train_timestep]).long().repeat(bsz).cuda()
                # timesteps = timesteps.long()

                encoder_hidden_states = text_encoder(batch["input_ids"])[0] # 1x 77 x 1024
                encoder_hidden_states_nshot = encoder_hidden_states.repeat(temp_nshot, 1, 1)

                target = -cond_latents_tag

                # Predict the noise residual and compute loss
                model_pred_cond_ref = unet(latents_rgb_cond_ref, timesteps, encoder_hidden_states_nshot, is_target=False).sample
                model_pred = unet(latents_tag, timesteps, encoder_hidden_states, is_target=True).sample
                if hasattr(unet, "module"):
                    unet.module.clear_attn_bank()
                else:
                    unet.clear_attn_bank()

                model_pred = model_pred.float() + model_pred_cond_ref.float() * 0.

                # if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if eval_results_all != {}:
                logs.update({str(str(dataset_name) + '_delta1'): eval_results_all[dataset_name]['delta1'] for dataset_name in eval_results_all.keys()})
                with open(args.output_dir + '/eval_results.txt', 'a') as f:
                    f.writelines(str(global_step) + ' , ')
                    for dataset_name in eval_results_all.keys():
                        for key in eval_results_all[dataset_name].keys():
                            f.writelines(dataset_name + '_' + key + ':' + str(eval_results_all[dataset_name][key]) + ' , ')
                    f.writelines('\n')
            progress_bar.set_postfix(**logs)

            # tensorboard to record the loss function
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    tracker.writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], global_step)
                    for dataset_name in eval_results_all.keys():
                        for key in eval_results_all[dataset_name].keys():
                            tracker.writer.add_scalar(str(dataset_name) + '_' + key, eval_results_all[dataset_name][key], global_step)
                else:
                    logger.warn(f"image logging not implemented for {tracker.name}")

            if global_step >= args.max_train_steps:
                break

            # if accelerator.is_main_process:
            if args.validation_images is not None and global_step % args.validation_steps == 0:
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = MarigoldPipelineRGBLatentNoise.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

        # Run a final round of inference.
        images = []
        if args.validation_prompts is not None:
            logger.info("Running inference for collecting generated images...")

            logger.info("Not implement, pass")
            pass

        if args.push_to_hub:
            save_model_card(args, repo_id, images, repo_folder=args.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()