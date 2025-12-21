# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Preprocess the Jigsaw dataset to parquet format
"""

import argparse
import os
import re
import json

import datasets

from verl.utils.hdfs_io import copy, makedirs

from PIL import Image


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default='/mnt/shared-storage-user/fangxinyu/jigsaw_project/RealJigsaw-RL/get_data/selected_4000_ids.json', help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="/mnt/shared-storage-user/fangxinyu/jigsaw_project/RealJigsaw-RL/get_data/train_data", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "unsplash/jigsaw"

    if local_dataset_path is not None:
        data_list = json.load(open(local_dataset_path))
        dataset = datasets.Dataset.from_dict({"id": data_list})
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_size = int(0.98 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))

    instruction_following = """
<image>You are an experienced Jigsaw Puzzle Master with exceptional visual reasoning skills.
Your task is to reconstruct a complete image from a 2x2 grid of mixed-up puzzle tiles.

### 1. INPUT DEFINITION
The user will provide an image containing 4 puzzle tiles. You must index these tiles based on their position in the provided image:
- **(1,1)**: Top-Left tile in the input image
- **(1,2)**: Top-Right tile in the input image
- **(2,1)**: Bottom-Left tile in the input image
- **(2,2)**: Bottom-Right tile in the input image

### 2. INSTRUCTIONS
**Step 1: Visual Reasoning**
Analyze the visual content inside `<Thinking>` tags.
* **Visual Continuity:** Look for continuous lines, matching colors, and completing objects across tile boundaries.
* **Global Context:** Identify the sky, ground, or main subjects to determine which tiles belong at the top vs. bottom, or left vs. right.
* **Logic:** Explain which tile connects to which based on the visual evidence.

**Step 2: Final Solution**
Determine the correct arrangement to form the complete image. The target order is row by row:
1. Top-Left position
2. Top-Right position
3. Bottom-Left position
4. Bottom-Right position

### 3. OUTPUT FORMAT
<Thinking>
[Provide your step-by-step visual reasoning here. Explain how the lines and colors connect.]
</Thinking>

<Final Answer>
(1,2), (2,1), (1,1), (2,2)
</Final Answer>

---

**NOW, SOLVE THE PUZZLE FOR THE PROVIDED IMAGE.**
"""

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # example is {'id': '...'}
            folder_id = example['id']
            root = '/mnt/shared-storage-user/mllm/fangxinyu/jigsaw/unsplash_train_data_22329/rect_and_grid_2*2'

            question = instruction_following
            
            folder_path = os.path.join(root, folder_id)
            # Assuming we use rect_pieces_meta.json
            meta_path = os.path.join(folder_path, "rect_pieces_meta.json")
            solution_meta_json = json.load(open(meta_path))
            solution_meta_json['root_path'] = folder_path
            
            # Define images - likely the grid board which contains the 4 tiles
            # Resize image to avoid DecompressionBombWarning and improve speed
            # Note: Original images are ~100MP (96M pixels), which causes hangs during dataset serialization.
            # Use thumbnail to resize while MAINTAINING ASPECT RATIO to avoid distortion.
            image = Image.open(os.path.join(folder_path, "rect_board.png")).convert("RGB")
            image.thumbnail((1024, 1024)) # Resizes to approx (1024, 671) preserving aspect ratio
            images = [image]

            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "images": images,
                "reward_model": {"style": "rule", "ground_truth": solution_meta_json},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "interaction_kwargs": {
                        "name": "jigsaw",
                        "query": question,
                        "ground_truth": solution_meta_json,
                    },
                },
            }
            return data

        return process_fn

    Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombWarning
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=4)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=4)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_save_dir
    # local_save_dir = args.local_dir
    # if local_save_dir is not None:
    #     print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    # else:
    #     local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train_w_iteraction_no_shape.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test_w_iteraction_no_shape.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
