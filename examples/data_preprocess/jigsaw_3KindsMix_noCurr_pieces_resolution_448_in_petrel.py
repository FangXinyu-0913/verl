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
import io

from petrel_client import client
cli_caption = client.Client("/mnt/shared-storage-user/fangxinyu/pluigins/caption_fxy.conf")


prompt_for_2x2 = """
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
Output the sequence of **Tile Indices** corresponding to the positions in raster-scan order (Row 1 left-to-right, then Row 2):
[
    Row 1: Col 1, Col 2,
    Row 2: Col 1, Col 2
]

---

**REQUIRED REASONING PROCESS (Do this inside <Thinking> tags):**

1. **Tile Analysis & Categorization:**
* **Corners:** Identify 4 tiles with *two* straight/flat edges. In a 2x2 grid, all tiles are corners.
* **Visual Features:** Note specific colors, lines, textures, or object parts (e.g., "half a face", "text snippet", "horizon line").


2. **Local Matching:** Find pairs of tiles that connect directly.
* *Texture continuity:* Does the grass pattern flow from Tile X to Tile Y?
* *Object continuity:* Does the car bumper in Tile A connect to the wheel in Tile B?


3. **Grid Assembly:** Place the Corner tiles into their correct positions (Top-Left, Top-Right, Bottom-Left, Bottom-Right) to establish the frame.
4. **Final Verification:** Ensure the sequence represents the raster-scan order (Top-Left to Bottom-Right).

---

**EXAMPLE OF DESIRED REASONING:**
<Thinking>
**Step 1: Categorization**

* **Corners (2 flat edges):**
* Tile 3 (Top & Left flat, shows Sun).
* Tile 1 (Top & Right flat, shows Blue Sky).
* Tile 4 (Bottom & Left flat, shows Grass).
* Tile 2 (Bottom & Right flat, shows Rock).

**Step 2: Feature Matching (Hypothetical Image: A Tree in a Field)**

* **Top Row Assembly:**
* Start with Corner Tile 3 (Top-Left Sun).
* Tile 3 has a cloud on the right edge. Tile 1 (Top-Right) has the rest of that cloud.
* *Row 1 Sequence: 3, 1.*

* **Bottom Row Assembly:**
* Tile 4 (Bottom-Left) shows the base of a tree.
* Tile 2 (Bottom-Right) shows the grass and a rock.
* Tile 4's right edge matches Tile 2's left edge.
* *Row 2 Sequence: 4, 2.*

**Step 3: Final Layout Construction**
Row 1: 3, 1
Row 2: 4, 2

**Final Check:**

* Corners are at all four positions.
* The tiles describe a picture of a sun above a tree in a field.
* All tiles are seamlessly fitted together; the interlocking contours of adjacent tiles match perfectly.

</Thinking>
<Final Answer>[3, 1, 4, 2]</Final Answer>

---

**NOW SOLVE THE PUZZLE:**
<Thinking>
[Your detailed step-by-step reasoning here]
</Thinking>
<Final Answer>[Your comma-separated sequence here]</Final Answer>
"""

prompt_for_3x3 = """
You are an experienced Jigsaw Puzzle Master with exceptional visual reasoning skills.
Your task is to reconstruct a complete PUZZLE from its individual tiles. The PUZZLE is divided into a 3 * 3 grid (total 9 tiles).

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

**YOUR GOAL:**
Determine the correct layout to reconstruct the original image.
Output the sequence of **Tile Indices** corresponding to the positions in raster-scan order (Row 1 left-to-right, then Row 2, etc.):
[
    Row 1: Col 1, Col 2, Col 3,
    Row 2: Col 1, Col 2, Col 3,
    Row 3: Col 1, Col 2, Col 3
]

---

**REQUIRED REASONING PROCESS (Do this inside <Thinking> tags):**

1. **Tile Analysis & Categorization:**
* **Corners:** Identify 4 tiles with *two* straight/flat edges.
* **Edges:** Identify 4 tiles with *one* straight/flat edge.
* **Interiors:** Identify 1 tile with *no* straight edges.
* **Visual Features:** Note specific colors, lines, textures, or object parts (e.g., "half a face", "text snippet", "horizon line").


2. **Local Matching:** Find pairs of tiles that connect directly.
* *Texture continuity:* Does the grass pattern flow from Tile X to Tile Y?
* *Object continuity:* Does the car bumper in Tile A connect to the wheel in Tile B?


3. **Grid Assembly:** Place the Corner tiles first to establish the frame, then fill in the Edge tiles, and finally the Interior tile.
4. **Final Verification:** Ensure the sequence represents the raster-scan order (Top-Left to Bottom-Right).

---

**EXAMPLE OF DESIRED REASONING:**
<Thinking>
**Step 1: Categorization**

* **Corners (2 flat edges):**
* Tile 7 (Top & Left flat, shows Sun).
* Tile 2 (Top & Right flat, shows Blue Sky).
* Tile 9 (Bottom & Left flat, shows Grass).
* Tile 4 (Bottom & Right flat, shows Rock).


* **Edges (1 flat edge):** Tiles 5, 8 (Top/Bottom edges); Tiles 1, 6 (Side edges).
* **Interiors:** Tile 3.

**Step 2: Feature Matching (Hypothetical Image: A Cabin in the Woods)**

* **Top Row Assembly:**
* Start with Corner Tile 7 (Top-Left Sun).
* Tile 7 has a cloud on the right edge. Tile 5 (Top Edge) has the rest of that cloud.
* Tile 5 connects to Corner Tile 2 (Top-Right Sky).
* *Row 1 Sequence: 7, 5, 2.*


* **Middle Row Assembly:**
* Below Tile 7 (Sun) is the forest line. Tile 1 (Side Edge) shows the treetops. It fits under 7.
* The central object is a Cabin. Tile 3 (Interior) shows the cabin roof and door. It fits to the right of Tile 1.
* Tile 6 (Side Edge) shows the right side of the cabin and more trees. Matches the right of Tile 3.
* *Row 2 Sequence: 1, 3, 6.*


* **Bottom Row Assembly:**
* Below Tile 1 is the ground. Tile 9 (Bottom-Left Corner) shows grass.
* Tile 8 (Bottom Edge) shows a path leading from the cabin. Matches right of Tile 9.
* Tile 4 (Bottom-Right Corner) shows the end of the path and a rock. Matches right of Tile 8.
* *Row 3 Sequence: 9, 8, 4.*


**Step 3: Final Layout Construction**
Row 1: 7, 5, 2
Row 2: 1, 3, 6
Row 3: 9, 8, 4

**Final Check:**

* Corners are at positions 1, 3, 7, 9 (in the raster-scan order).
* The tiles describe a coherent picture of a cabin in the woods under a sunny sky.
* All tiles are seamlessly fitted together; the interlocking visual content of adjacent tiles matches perfectly.

</Thinking>
<Final Answer>[7, 5, 2, 1, 3, 6, 9, 8, 4]</Final Answer>

---

**NOW SOLVE THE PUZZLE:**
<Thinking>
[Your detailed step-by-step reasoning here]
</Thinking>
<Final Answer>[Your comma-separated sequence here]</Final Answer>
"""

prompt_for_4x4 = """
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

prompt_for_mix = {
    "2*2": prompt_for_2x2,
    "3*3": prompt_for_3x3,
    "4*4": prompt_for_4x4
}


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

    data_source = "unsplash/jigsaw_mix"

    if local_dataset_path is not None:
        data_list = json.load(open(local_dataset_path))
        dataset = datasets.Dataset.from_dict({"id": data_list})
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_size = int(0.98 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))
    

    # 创建处理单个puzzle类型的函数
    def make_single_puzzle_map_fn(split, puzzle_type):
        def process_fn(example, idx):
            # example is {'id': '...'}
            folder_id = example['id']
            
            # 根据puzzle_type设置不同的root路径
            root = f's3://fangxinyu/jigsaw_data/unsplash_train_data_22329_rect3kinds_251225/rect_and_grid_{puzzle_type}/'
            
            folder_path = root + folder_id
            
            
            # Assuming we use rect_pieces_meta.json
            meta_path = os.path.join(folder_path, "rect_pieces_meta.json")
                
            # solution_meta_json = json.load(open(meta_path))
            solution_meta_json = json.loads(cli_caption.get(meta_path))
            solution_meta_json['root_path'] = folder_path
            
            # 根据puzzle_type确定board_id列表
            if puzzle_type == '2*2':
                board_ids = ["1", "2", "3", "4"]
                grid_size = "2 * 2"
                total_tiles = 4
            elif puzzle_type == '3*3':
                board_ids = [str(i) for i in range(1, 10)]  # 1-9
                grid_size = "3 * 3"
                total_tiles = 9
            elif puzzle_type == '4*4':
                board_ids = [str(i) for i in range(1, 17)]  # 1-16
                grid_size = "4 * 4"
                total_tiles = 16
            else:
                return {}
            
            question = prompt_for_mix[puzzle_type]
            
            board_id_to_filename = {}
            for piece in solution_meta_json['pieces']:
                board_id_to_filename[str(piece['board_id'])] = piece['filename']
            solution_meta_json['board_id_to_filename'] = board_id_to_filename

            images = []
            for board_id in board_ids:
                if board_id not in board_id_to_filename:
                    continue
                image = Image.open(io.BytesIO(cli_caption.get(os.path.join(folder_path, board_id_to_filename[board_id])))).convert("RGB")
                image.thumbnail((448, 448)) # Resizes to approx (1024, 671) preserving aspect ratio
                images.append(image)
            
            # 如果图片数量不匹配，跳过这条数据
            if len(images) != total_tiles:
                print(f"Skipping data {folder_id} because the number of images ({len(images)}) does not match the number of tiles ({total_tiles})")
                return {}
            
            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "images": images,
                "reward_model": {"style": "rule", "ground_truth": solution_meta_json},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "original_idx": idx,  # 保存原始索引用于排序
                    "puzzle_type": puzzle_type,  # 添加puzzle_type标识
                    "need_tools_kwargs": False,
                },
            }
            return data
        
        return process_fn

    Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombWarning
    
    # 将train_dataset分成三份，每份大小相同
    # third_size = train_dataset_size // 3
    # train_part1 = train_dataset.select(range(0, third_size))
    # train_part2 = train_dataset.select(range(third_size, 2 * third_size))
    # train_part3 = train_dataset.select(range(2 * third_size, train_dataset_size))
    
    # 分别处理三种puzzle类型
    print("Processing 2*2 puzzles...")
    train_2x2 = train_dataset.map(function=make_single_puzzle_map_fn("train", "2*2"), with_indices=True, num_proc=10)
    test_2x2 = test_dataset.map(function=make_single_puzzle_map_fn("test", "2*2"), with_indices=True, num_proc=10)
    
    print("Processing 3*3 puzzles...")
    train_3x3 = train_dataset.map(function=make_single_puzzle_map_fn("train", "3*3"), with_indices=True, num_proc=10)
    test_3x3 = test_dataset.map(function=make_single_puzzle_map_fn("test", "3*3"), with_indices=True, num_proc=10)

    print("Processing 4*4 puzzles...")
    train_4x4 = train_dataset.map(function=make_single_puzzle_map_fn("train", "4*4"), with_indices=True, num_proc=10)
    test_4x4 = test_dataset.map(function=make_single_puzzle_map_fn("test", "4*4"), with_indices=True, num_proc=10)
    # 按照比例重组数据
    # 40% 2*2，30% 3*3，30% 4*4，形成一个数据集即可
    train_2x2_count = int(4000 * 0.98 * 0.4)
    train_3x3_count = int(4000 * 0.98 * 0.3)
    train_4x4_count = int(4000 * 0.98 * 0.3)
    
    
    
    # 随机挑选并合并三个部分
    train_dataset = datasets.concatenate_datasets([train_2x2.shuffle(seed=42).select(range(train_2x2_count)), train_3x3.shuffle(seed=42).select(range(train_3x3_count)), train_4x4.shuffle(seed=42).select(range(train_4x4_count))])

    print("Shuffling each part...")
    test_2x2 = test_2x2.shuffle(seed=42)
    test_3x3 = test_3x3.shuffle(seed=42)
    test_4x4 = test_4x4.shuffle(seed=42)
    
    test_dataset = datasets.concatenate_datasets([test_2x2, test_3x3, test_4x4])

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train_3KindsMix_pieces_resolution448_OneStage.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test_3KindsMix_pieces_resolution448_OneStage.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
