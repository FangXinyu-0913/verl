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

    data_source = "unsplash/jigsaw_output_coordinate_dict"

    if local_dataset_path is not None:
        data_list = json.load(open(local_dataset_path))
        dataset = datasets.Dataset.from_dict({"id": data_list})
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_size = int(0.98 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))

    instruction_following = """
You are an experienced Jigsaw Puzzle Master with exceptional visual reasoning skills.
Your task is to reconstruct a complete PUZZLE from its individual tiles. The PUZZLE is divided into a 2 * 2 grid (total 4 tiles).

**INPUT TILES:**
The following images are the individual tiles of the puzzle. Each tile is assigned a numeric Index.
Note that each tile is in its correct orientation and has not been rotated.
  - **Index 1:** <image>
  - **Index 2:** <image>
  - **Index 3:** <image>
  - **Index 4:** <image>

**YOUR GOAL:**
Determine the correct layout to reconstruct the original image.
Output the sequence of Tile Indices corresponding to the positions in raster-scan order:
[Top-Left, Top-Right, Bottom-Left, Bottom-Right]

-----

**REQUIRED REASONING PROCESS (Do this inside <Thinking> tags):**

1.  **Tile Analysis:** Briefly describe the visual content of each tile. Refer to tiles by their Index (e.g., Tile 1). Look for distinctive features like edges, lines, colors, or parts of objects.
2.  **Edge Matching:** Identify connections between tiles.
      - *Horizontal match:* Does the right edge of one tile match the left edge of another?
      - *Vertical match:* Does the bottom edge of one tile match the top edge of another?
3.  **Global Consistency:** Check if the arrangement follows physical logic (e.g., sky is usually at the top, ground at the bottom).
4.  **Final Sequence:** Assemble the indices.

-----

**EXAMPLE OF DESIRED REASONING:**
<Thinking>
**Tile Analysis:**

  - Tile 1: Shows blue sky and the top of a tree.
  - Tile 2: Shows grass and the roots of a tree.
  - Tile 3: Shows the trunk of the tree and horizon line.
  - Tile 4: Shows a bird flying in clouds.

**Edge Matching:**

  - Tile 1 (Sky/Tree Top) seems to fit above Tile 3 (Trunk).
  - Tile 4 (Bird/Clouds) is also sky, likely fits next to Tile 1. Let's check the cloud patterns... Yes, Tile 4 is the top-right.
  - Tile 3 (Trunk) connects down to Tile 2 (Roots).

**Layout Proposal:**
Row 1: Tile 1 (Left), Tile 4 (Right)
Row 2: Tile 3 (Left), Tile 2 (Right)

**Final Check:**
Sky is up, grass is down. Tree trunk connects top to bottom.
The sequence is 1, 4, 3, 2.
</Thinking>
<Final Answer>1,4,3,2</Final Answer>

-----

**NOW SOLVE THE PUZZLE:**
<Thinking>
[Your step-by-Step reasoning here]
</Thinking>
<Final Answer>[Your comma-separated sequence here]</Final Answer>
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
            # image = Image.open(os.path.join(folder_path, "rect_board.png")).convert("RGB")
            board_id_to_filename = {}
            for item in solution_meta_json['pieces']:
                board_id_to_filename[item['board_id']] = item['filename']

            # print(f'board_id_to_filename: {board_id_to_filename}, folder_path: {folder_path}')
            
            images = []
            for board_id in [1, 2, 3, 4]:
                image = Image.open(os.path.join(folder_path, board_id_to_filename[board_id])).convert("RGB")
                image.thumbnail((1024, 1024)) # Resizes to approx (1024, 671) preserving aspect ratio
                images.append(image)
            
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
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_save_dir
    # local_save_dir = args.local_dir
    # if local_save_dir is not None:
    #     print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    # else:
    #     local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train_four_pieces_no_shape.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test_four_pieces_no_shape.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
