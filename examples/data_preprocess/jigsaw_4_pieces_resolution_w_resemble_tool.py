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

    data_source = "unsplash/jigsaw_with_tool"

    if local_dataset_path is not None:
        data_list = json.load(open(local_dataset_path))
        dataset = datasets.Dataset.from_dict({"id": data_list})
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_size = int(0.98 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))

    instruction_following = """
You are an experienced Jigsaw Puzzle Master with exceptional visual reasoning skills and geometric awareness.
Your task is to reconstruct a complete PUZZLE from its individual tiles. The PUZZLE is divided into a 2 * 2 grid (total 4 tiles).

**INPUT IMAGES:**
The following images represent the individual PUZZLE tiles. Each tile is assigned a numeric Index.
Note that each tile is in its correct orientation and has not been rotated.
  - Index 1: <image>
  - Index 2: <image>
  - Index 3: <image>
  - Index 4: <image>

**YOUR GOAL:**
Execute the following three steps sequentially to solve the puzzle:

1.  **Shape Extraction:** Analyze the geometry of each tile. For every tile from Index 1 to Index 4, describe the shape of its four edges in the specific order: **Left, Top, Right, Bottom**.

      * Possible shapes are: **straight**, **innie**, **outie**.
      * Enclose this analysis inside `<Shape></Shape>` tags.

2.  **Reasoning & Matching:** Follow the "REQUIRED REASONING PROCESS" below, detailedly analyzing visual content and matching the edge shapes identified in Step 1. Enclose your thought process inside `<Thinking></Thinking>` tags.

3.  **Reconstruction:** Determine the correct layout to reconstruct the original image. Output the sequence of Tile Indices corresponding to the final positions in raster-scan order: [Top-Left, Top-Right, Bottom-Left, Bottom-Right].

-----

**REQUIRED REASONING PROCESS (Do this inside <Thinking> tags):**

1.  **Tile Analysis:** Briefly describe the visual content of each tile (e.g., colors, objects, lines).
2.  **Edge Matching:** Identify connections between tiles using both **Visual Features** (lines crossing borders) and **Geometric Compatibility** (from your Shape Extraction):
      * *Horizontal match:* Does the right edge of one tile (e.g., outie) fit the left edge of another (e.g., innie)?
      * *Vertical match:* Does the bottom edge of one tile fit the top edge of another?
3.  **Global Consistency:** Check if the arrangement follows physical logic (e.g., sky is top, ground is bottom).
4.  **Final Sequence:** Assemble the indices.

-----

**EXAMPLE OF DESIRED REASONING:**

<Shape>
Tile 1: Left=straight, Top=straight, Right=outie, Bottom=innie
Tile 2: Left=outie, Top=innie, Right=straight, Bottom=straight
Tile 3: Left=straight, Top=outie, Right=innie, Bottom=straight
Tile 4: Left=innie, Top=straight, Right=straight, Bottom=outie
</Shape>

<Thinking>
**Tile Analysis:**

  - Tile 1: Shows blue sky and the top of a tree.
  - Tile 2: Shows grass and the roots of a tree.
  - Tile 3: Shows the trunk of the tree and horizon line.
  - Tile 4: Shows a bird flying in clouds.

**Edge Matching:**

  - **Geometry Check:** Tile 1 has Left=straight and Top=straight, so it must be the Top-Left corner. Its Bottom is "innie".
  - Tile 3 has Top="outie". This geometrically fits into the bottom of Tile 1.
  - **Visual Check:** Tile 1 (Tree Top) visually connects above Tile 3 (Trunk).
  - Tile 4 has Top=straight and Right=straight, making it the Top-Right corner. Its Left is "innie", which fits Tile 1's Right "outie".

**Layout Proposal:**
Row 1: Tile 1 (Left), Tile 4 (Right)
Row 2: Tile 3 (Left), Tile 2 (Right)

**Final Check:**
Sky is up, grass is down. Tree trunk connects top to bottom. Shapes interlock correctly.
The sequence is 1, 4, 3, 2.
</Thinking>

<Final Answer>1,4,3,2</Final Answer>

-----

**NOW SOLVE THE PUZZLE:**
<Shape>
[Your edge shape analysis here]
</Shape>
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
                board_id_to_filename[str(item['board_id'])] = item['filename']
            solution_meta_json['board_id_to_filename'] = board_id_to_filename

            # print(f'board_id_to_filename: {board_id_to_filename}, folder_path: {folder_path}')
            
            images = []
            for board_id in ["1", "2", "3", "4"]:
                image = Image.open(os.path.join(folder_path, board_id_to_filename[board_id])).convert("RGB")
                image.thumbnail((448, 448)) # Resizes to approx (1024, 671) preserving aspect ratio
                images.append(image)
            
            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are an experienced Jigsaw Puzzle Master with exceptional visual reasoning skills. "
                            "You are given a jigsaw puzzle and you need to solve it step by step. "
                            "Reasoning step by step before any tool call. "
                            "You can use the `jigsaw_restoration_tool` tool after step by step reasoning to restore the puzzle based on your answer, "
                            "fully utilize these tools to generate final answer and refine your answer if necessary. "
                        ),
                    },
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
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "jigsaw_restoration_tool": {
                            "create_kwargs": {"ground_truth": solution_meta_json},
                        },
                    },
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
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=10)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=10)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_save_dir
    # local_save_dir = args.local_dir
    # if local_save_dir is not None:
    #     print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    # else:
    #     local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train_four_pieces_resolution448_w_resemble_tool.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test_four_pieces_resolution448_w_resemble_tool.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
