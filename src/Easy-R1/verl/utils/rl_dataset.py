# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# limitations under the License.

import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_5_vl import get_rope_index
from PIL import Image


def collate_fn(features: List[Dict[str, Any]], world_size=8) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feat in features:
        for feature in feat:
            for key, value in feature.items():
                if isinstance(value, torch.Tensor):
                    tensors[key].append(value)
                else:
                    non_tensors[key].append(value)
    batch_size = len(tensors["input_ids"])
    remainder = batch_size % world_size
    pad_len = (world_size - remainder) % world_size

    for i in range(pad_len):
        for key in tensors:
            tensors[key].append(tensors[key][i % remainder])
        
        for key in non_tensors:
            non_tensors[key].append(non_tensors[key][i % remainder])

    for key, value in tensors.items():
        if key not in ["pixel_values", "image_grid_thw"]:
            tensors[key] = torch.stack(value, dim=0)

    final_batch_size = tensors["input_ids"].shape[0]
    return {**tensors, **non_tensors}


def process_image(image_path: ImageObject, max_pixels: int, min_pixels: int) -> ImageObject:
    image = Image.open(image_path).convert('RGB')
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key="prompt",
        max_prompt_length=1024,
        truncation="error",
        system_prompt=None,
        max_pixels=None,
        min_pixels=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        # self.dataset = load_dataset(data_path, split=data_split)
        self.dataset = []
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.dataset.append(data)

    def __len__(self):
        print("length: ", len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_list = []
        row_base = self.dataset[index]
        for row_dict in row_base['steps']:
            # row_dict = self.dataset[index]
            messages = [
                {"role": "system", "content": row_dict["system"]},
                {"role": "user", "content": row_dict[self.prompt_key]},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            dummy = False
            if "images" in row_base and len(row_base['images']) > 0:  # expand image token
                raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
                row_dict["images"] = [
                    process_image(image, self.max_pixels, self.min_pixels) for image in row_base["images"]
                ]
                image_inputs = self.processor.image_processor(row_dict["images"], return_tensors="pt")
                image_grid_thw = image_inputs["image_grid_thw"]
                row_dict.update(image_inputs)
                # if image_grid_thw.sum() == 0:
                #     raise ValueError
                if image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size**2
                    index = 0
                    while "<image>" in prompt:
                        prompt = prompt.replace(
                            "<image>",
                            "<|vision_start|>"
                            + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                            + "<|vision_end|>",
                            1,
                        )
                        index += 1

                    prompt = prompt.replace("<|placeholder|>", self.processor.image_token)
            else:
                dummy = True
                raw_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + prompt
                dummy_image_array = np.uint8(np.ones((224,224,3))*255)  
                dummy_image = Image.fromarray(dummy_image_array)
                image_inputs = self.processor.image_processor(dummy_image, return_tensors="pt")
                image_grid_thw = image_inputs["image_grid_thw"]
                # if image_grid_thw.sum() == 0:
                #     raise ValueError
                
                row_dict["images"] = [dummy_image]
                row_dict.update(image_inputs)

                merge_length = self.processor.image_processor.merge_size ** 2
                dummy_placeholder_count = image_grid_thw.prod() // merge_length
                dummy_image_placeholder = "<|vision_start|>" + "<|placeholder|>" * dummy_placeholder_count + "<|vision_end|>"
                prompt = dummy_image_placeholder + prompt
                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)
                dummy_img_token_length = len(
                    self.tokenizer(dummy_image_placeholder, add_special_tokens=False).input_ids
                )

            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt,
                tokenizer=self.tokenizer,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )

            if dummy:
                attention_mask[:dummy_img_token_length] = 0

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )  # (3, seq_len)
            # if not dummy:
            #     position_ids = get_rope_index(
            #         self.processor,
            #         input_ids=input_ids,
            #         image_grid_thw=image_grid_thw,
            #         attention_mask=attention_mask,
            #     )  # (3, seq_len)
            # else:
            #     
            #     position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seqlen,)
                # position_ids = position_ids.unsqueeze(0).repeat(3, 1)
                # position_ids = torch.stack([
                #     position_ids,
                #     torch.zeros_like(position_ids),
                #     torch.zeros_like(position_ids)
                # ], dim=0)
            row_dict["input_ids"] = input_ids
            row_dict["attention_mask"] = attention_mask
            row_dict["position_ids"] = position_ids
            row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            row_list.append(row_dict)
        
        return row_list
