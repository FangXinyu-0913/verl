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
"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os

import datasets
import json
from verl.utils.hdfs_io import copy, makedirs
from PIL import Image

def not_empty(example):
    return example != {}

def valid_image(example):
    image_path = example["image"]
    try:
        Image.open(os.path.join(args.original_dir, image_path), "r")
        return True
    except Exception:
        print(f"can not open {os.path.join(args.original_dir, image_path)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/chart2code_mimic")
    parser.add_argument("--original_dir", default="/fs-computility/mllm1/fangxinyu/plot2code/ChartMimic/dataset")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "ChartMimic/ChartMimic"

    dataset = {}
    dataset['train'] = json.load(open(os.path.join(args.original_dir,'train_direct_mimic.json'),'r'))
    dataset['test'] = json.load(open(os.path.join(args.original_dir,'test_direct_mimic.json'),'r'))

    train_dataset_lst = dataset["train"]
    test_dataset_lst = dataset["test"]

    from datasets import Dataset

    # 先从你的 list 创建 Dataset 对象
    train_dataset = Dataset.from_list(train_dataset_lst)
    test_dataset = Dataset.from_list(test_dataset_lst)


    # 过滤掉打不开的图片
    train_dataset = train_dataset.filter(valid_image, num_proc=8)
    test_dataset = test_dataset.filter(valid_image, num_proc=8)

    def make_map_fn(split):
        def process_fn(example, idx):
            conv = example['conversations']
            prompt = [x['value'] for x in conv if x['from'] == 'human'][0]
            answer = [x['value'] for x in conv if x['from'] == 'gpt'][0]


            image_path = example.pop("image")
            try:
                image = Image.open(os.path.join(args.original_dir, image_path), "r")
            except Exception as e:
                print(f'can not open {os.path.join(args.original_dir, image_path)}')
                return {}


            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": [image],
                "ability": "code",
                "reward_model": {
                    "style": "rule", 
                    "ground_truth": {
                        "answer": answer,
                        "index": idx
                    }
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "orignal_index": example['id'],
                    "answer": answer,
                    "question": prompt,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
