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
        "--local_save_dir", default="/mnt/shared-storage-user/mllm/fangxinyu/jigsaw/train_data_for_verl", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "unsplash/jigsaw_sixteen"

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
Your task is to reconstruct a complete PUZZLE from its individual tiles. The PUZZLE is divided into a 4 * 4 grid (total 16 tiles).

**INPUT TILES:**
The following images are the individual tiles of the puzzle. Each tile is assigned a numeric Index.
Note that each tile is in its correct orientation and has not been rotated.
  - **Index 1:** <image>
  - **Index 2:** <image>
  - **Index 3:** <image>
  - **Index 4:** <image>
  - **Index 5:** <image>
  - **Index 6:** <image>
  - **Index 7:** <image>
  - **Index 8:** <image>
  - **Index 9:** <image>
  - **Index 10:** <image>
  - **Index 11:** <image>
  - **Index 12:** <image>
  - **Index 13:** <image>
  - **Index 14:** <image>
  - **Index 15:** <image>
  - **Index 16:** <image>

**YOUR GOAL:**
Determine the correct layout to reconstruct the original image.
Output the sequence of **Tile Indices** corresponding to the positions in raster-scan order (Row 1 left-to-right, then Row 2, etc.):
[
    Row 1: Col 1, Col 2, Col 3, Col 4,
    Row 2: Col 1, Col 2, Col 3, Col 4,
    Row 3: Col 1, Col 2, Col 3, Col 4,
    Row 4: Col 1, Col 2, Col 3, Col 4
]

---

**REQUIRED REASONING PROCESS (Do this inside <Thinking> tags):**

1. **Tile Analysis & Categorization:**
* **Corners:** Identify 4 tiles with *two* straight/flat edges.
* **Edges:** Identify 8 tiles with *one* straight/flat edge.
* **Interiors:** Identify 4 tiles with *no* straight edges.
* **Visual Features:** Note specific colors, lines, textures, or object parts (e.g., "half a face", "text snippet", "horizon line").


2. **Local Matching:** Find pairs of tiles that connect directly.
* *Texture continuity:* Does the grass pattern flow from Tile X to Tile Y?
* *Object continuity:* Does the car bumper in Tile A connect to the wheel in Tile B?


3. **Grid Assembly:** Place the Corner tiles first to establish the frame, then fill in the Edge tiles, and finally the Interior tiles.
4. **Final Verification:** Ensure the sequence represents the raster-scan order (Top-Left to Bottom-Right).

---

**EXAMPLE OF DESIRED REASONING:**
<Thinking>
**Step 1: Categorization**

* **Corners (2 flat edges):**
* Tile 14 (Top & Left flat, shows Sun).
* Tile 3 (Top & Right flat, shows Blue Sky).
* Tile 9 (Bottom & Left flat, shows Grass).
* Tile 5 (Bottom & Right flat, shows Rock).


* **Edges (1 flat edge):** Tiles 2, 8, 11, 15 (Top/Bottom edges); Tiles 1, 6, 12, 16 (Side edges).
* **Interiors:** Tiles 4, 7, 10, 13.

**Step 2: Feature Matching (Hypothetical Image: A Dog in a Park)**

* **Top Row Assembly:**
* Start with Corner Tile 14 (Top-Left Sun).
* Tile 14 has a cloud on the right edge. Tile 8 (Top Edge) has the rest of that cloud.
* Tile 8 connects to Tile 11 (half of the blue sky, 8 right edge is outie and 11 left edge is innie, both shows the blue sky).
* Tile 11 connects to Corner Tile 3 (Top-Right Sky).
* *Row 1 Sequence: 14, 8, 11, 3.*


* **Middle Rows Assembly:**
* Below Tile 14 (Sun) is the horizon. Tile 1 (Side Edge) shows the horizon line and mountains. It fits under 14.
* The central object is a Dog. Tile 7 shows the dog's head. It fits to the right of Tile 1.
* Tile 10 shows the ball the dog want to catch. Matches the right of Tile 7.
* ... [Continuing logic for remaining pieces] ...


**Step 3: Final Layout Construction**
Row 1: 14, 8, 11, 3
Row 2: 1, 7, 10, 6
Row 3: 12, 4, 13, 16
Row 4: 9, 2, 15, 5

**Final Check:**

* Corners are at positions 1, 4, 13, 16 (in the 1D list indices).
* The middle tiles describe the picture that a dog want to catch the ball in vivid.
* All tiles are seamlessly fitted together; the interlocking contours of adjacent tiles match perfectly.

</Thinking>
<Final Answer>[14, 8, 11, 3, 1, 7, 10, 6, 12, 4, 13, 16, 9, 2, 15, 5]</Final Answer>

---

**NOW SOLVE THE PUZZLE:**
<Thinking>
[Your detailed step-by-step reasoning here]
</Thinking>
<Final Answer>[Your comma-separated sequence here]</Final Answer>
"""

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # example is {'id': '...'}
            try:
                folder_id = example['id']
                root = '/mnt/shared-storage-user/mllm/fangxinyu/jigsaw/unsplash_train_data_22329/rect_and_grid_4*4'

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
                for board_id in range(1, 17):
                    image = Image.open(os.path.join(folder_path, board_id_to_filename[str(board_id)])).convert("RGB")
                    image.thumbnail((448, 448)) # Resizes to approx (1024, 671) preserving aspect ratio
                    images.append(image)
                print(len(images))
                
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
            except Exception as e:
                print(f'Error processing example {idx}: {e}')
                # 必须返回具有相同结构的字典，否则会报 KeyError
                # 我们用空列表 [] 或空内容标记这条数据无效
                data = {
                    "data_source": data_source,  # 必须包含这个 key
                    "agent_name": "tool_agent",
                    "prompt": [],
                    "images": [],                # 关键：用空列表标记失败
                    "reward_model": {},
                    "extra_info": {}
                }
            return data

        return process_fn

    Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombWarning
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=20)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=20)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_save_dir
    # local_save_dir = args.local_dir
    # if local_save_dir is not None:
    #     print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    # else:
    #     local_save_dir = args.local_save_dir

    print(f"Before filter: Train {len(train_dataset)}, Test {len(test_dataset)}")
    train_dataset = train_dataset.filter(lambda x: len(x['images']) > 0, num_proc=12)
    test_dataset = test_dataset.filter(lambda x: len(x['images']) > 0, num_proc=12)
    print(f"After filter: Train {len(train_dataset)}, Test {len(test_dataset)}")

    train_dataset.to_parquet(os.path.join(local_save_dir, "train_16_pieces_no_shape_resolution448.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test_16_pieces_no_shape_resolution448.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
