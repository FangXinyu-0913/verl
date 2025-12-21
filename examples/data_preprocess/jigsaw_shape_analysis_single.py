# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# # Copyright 2023-2024 SGLang Team
# # Copyright 2025 ModelBest Inc. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """
# Preprocess the Jigsaw dataset to parquet format
# """

# import argparse
# import os
# import re
# import json

# import datasets
# from verl.utils.hdfs_io import copy, makedirs
# from PIL import Image

# # Disable DecompressionBombWarning globally
# Image.MAX_IMAGE_PIXELS = None 

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--hdfs_dir", default=None)
#     parser.add_argument("--local_dataset_path", default='/mnt/shared-storage-user/fangxinyu/jigsaw_project/RealJigsaw-RL/get_data/selected_4000_ids.json', help="The local path to the raw dataset, if it exists.")
#     parser.add_argument(
#         "--local_save_dir", default="/mnt/shared-storage-user/fangxinyu/jigsaw_project/RealJigsaw-RL/get_data/train_data", help="The save directory for the preprocessed dataset."
#     )

#     args = parser.parse_args()
#     local_dataset_path = args.local_dataset_path

#     data_source = "unsplash/jigsaw_shape_analysis_singlePieceRect"

#     if local_dataset_path is not None:
#         data_list = json.load(open(local_dataset_path))
#         dataset = datasets.Dataset.from_dict({"id": data_list})
#     else:
#         dataset = datasets.load_dataset(data_source, "main")

#     train_size = int(0.98 * len(dataset))
#     train_dataset = dataset.select(range(train_size))
#     test_dataset = dataset.select(range(train_size, len(dataset)))

#     instruction_following = """
# <image>You are an experienced Jigsaw Puzzle Analyzer with exceptional visual perception.
# Your task is to identify the edge shapes of a **single puzzle tile**.

# ### 1. INPUT DEFINITION
# The user will provide an image containing **one** puzzle tile.
# You must analyze this tile relative to the image frame (Up is Top).

# ### 2. ANALYSIS STEPS

# **Step 1: Visual Observation**
# Inside `<Thinking>` tags, inspect the tile's 4 edges in **Clockwise Order** (Top -> Right -> Bottom -> Left).
# - Determine strictly if each edge is:
#     - `'flat'`: A straight edge (border/frame).
#     - `'innie'`: A concave shape (hole/socket).
#     - `'outie'`: A convex shape (tab/head).
#     - `'unknown'`: If the edge is unclear or obscured.

# **Step 2: Shape Analysis**
# Inside `<Shape Analysis>` tags, output ONLY the shape definition tuple.
# - **Order:** Top -> Right -> Bottom -> Left.
# - **Vocabulary:** Use exactly: `'flat'`, `'innie'`, `'outie'`, `'unknown'`.
# - **Format:** A single Python tuple containing 4 strings. Do NOT use dictionaries or other text.

# ### 3. OUTPUT FORMAT EXAMPLE

# <Thinking>
# Looking at the single tile:
# - The Top edge is perfectly straight.
# - The Right edge has a protruding tab (outie).
# - The Bottom edge has a hole indentation (innie).
# - The Left edge is also straight.
# </Thinking>

# <Shape Analysis>
# ('flat', 'outie', 'innie', 'flat')
# </Shape Analysis>
# """

#     def make_map_fn(split):
#         def process_fn(examples):
#             # Because we use batched=True, 'examples' is a dict of lists: {'id': ['id1', 'id2', ...]}
            
#             # Prepare lists to store the exploded data
#             new_data_sources = []
#             new_prompts = []
#             new_images = []
#             new_reward_models = []
#             new_extra_infos = []

#             root = '/mnt/shared-storage-user/mllm/fangxinyu/jigsaw/unsplash_train_data_22329/rect_and_grid_2*2'
            
#             # Iterate over the batch
#             for idx, folder_id in enumerate(examples['id']):
#                 folder_path = os.path.join(root, folder_id)
#                 meta_path = os.path.join(folder_path, "rect_pieces_meta.json")
                
#                 # Load the meta data containing pieces info
#                 solution_meta_json = json.load(open(meta_path))
#                 pieces = solution_meta_json["pieces"]
                
#                 # Load the main board image once per folder
#                 # board_image_path = os.path.join(folder_path, "rect_board.png")
#                 # board_image = Image.open(board_image_path).convert("RGB")
                
#                 # Iterate through the 4 pieces in the meta json
#                 for piece_idx, piece_info in enumerate(pieces):

                    
#                     # --- Construct Data Item ---
#                     new_data_sources.append(data_source)
                    
#                     new_prompts.append([
#                         {
#                             "role": "user",
#                             "content": instruction_following,
#                         }
#                     ])
                    
#                     image = Image.open(os.path.join(folder_path, piece_info['filename'])).convert("RGB")
#                     image.thumbnail((256, 256)) # Resizes to approx (1024, 671) preserving aspect ratio
#                     images = [image]
#                     new_images.append(images)
                    
#                     # Store piece-specific ground truth
#                     new_reward_models.append({
#                         "style": "rule", 
#                         "ground_truth": piece_info  # Passing the specific piece dict
#                     })
                    
#                     new_extra_infos.append({
#                         "split": split,
#                         "original_folder_id": folder_id,
#                         "piece_index": piece_idx,
#                         "board_id": piece_info['board_id'],
#                         "interaction_kwargs": {
#                             "name": "jigsaw_single_piece",
#                             "query": instruction_following,
#                             "ground_truth": piece_info,
#                         },
#                     })

#             return {
#                 "data_source": new_data_sources,
#                 "prompt": new_prompts,
#                 "images": new_images,
#                 "reward_model": new_reward_models,
#                 "extra_info": new_extra_infos
#             }

#         return process_fn

#     # Use batched=True to allow 1 input row -> N output rows
#     # remove_columns=['id'] ensures the old columns don't conflict with new dictionary keys if names differ,
#     # and handles the size mismatch (N inputs vs 4N outputs)
    
#     train_dataset = train_dataset.map(
#         function=make_map_fn("train"), 
#         batched=True, 
#         batch_size=10, 
#         remove_columns=dataset.column_names,
#         num_proc=4
#     )
    
#     test_dataset = test_dataset.map(
#         function=make_map_fn("test"), 
#         batched=True, 
#         batch_size=10, 
#         remove_columns=dataset.column_names,
#         num_proc=4
#     )

#     hdfs_dir = args.hdfs_dir
#     local_save_dir = args.local_save_dir

#     if not os.path.exists(local_save_dir):
#         os.makedirs(local_save_dir, exist_ok=True)

#     print(f"Saving train dataset with {len(train_dataset)} items...")
#     train_dataset.to_parquet(os.path.join(local_save_dir, "train_only_shape_analysis_single_only_rect.parquet"))
    
#     print(f"Saving test dataset with {len(test_dataset)} items...")
#     test_dataset.to_parquet(os.path.join(local_save_dir, "test_only_shape_analysis_single_only_rect.parquet"))

#     if hdfs_dir is not None:
#         makedirs(hdfs_dir)
#         copy(src=local_save_dir, dst=hdfs_dir)



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
import random
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default='/mnt/shared-storage-user/fangxinyu/jigsaw_project/RealJigsaw-RL/get_data/selected_4000_ids.json', help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="/mnt/shared-storage-user/fangxinyu/jigsaw_project/RealJigsaw-RL/get_data/train_data", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "unsplash/jigsaw_shape_analysis_single"

    if local_dataset_path is not None:
        data_list = json.load(open(local_dataset_path))
        dataset = datasets.Dataset.from_dict({"id": data_list})
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_size = int(0.98 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))

    instruction_following = """
<image>**Role:** Jigsaw Puzzle Analyzer
**Task:** Identify the edge shapes of the single puzzle tile in the image.

**Allowed Vocabulary:**

  - `"flat"`: A straight edge (border).
  - `"outie"`: A protruding tab.
  - `"innie"`: An indented hole.

**Instructions:**

1.  Analyze the tile edges relative to the image frame (Up is Top).
2.  Inspect edges in **Clockwise Order**: Top -> Right -> Bottom -> Left.
3.  Output **ONLY** a raw JSON object. Do not use markdown or explanations.

**Output Example:**
{
  "top": "flat",
  "right": "outie",
  "bottom": "innie",
  "left": "flat"
}
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

            selected_piece = random.choice(solution_meta_json['pieces'])
            
            # Define images - likely the grid board which contains the 4 tiles
            # Resize image to avoid DecompressionBombWarning and improve speed
            # Note: Original images are ~100MP (96M pixels), which causes hangs during dataset serialization.
            # Use thumbnail to resize while MAINTAINING ASPECT RATIO to avoid distortion.
            image = Image.open(os.path.join(folder_path, selected_piece['filename'])).convert("RGB")
            image.thumbnail((1024, 1024)) # Resizes to approx (1024, 671) preserving aspect ratio
            images = [image]

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "images": images,
                "reward_model": {"style": "rule", "ground_truth": selected_piece},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "folder_id": folder_id,
                    "interaction_kwargs": {
                        "name": "jigsaw",
                        "query": question,
                        "ground_truth": selected_piece,
                    },
                },
            }

            print(folder_path, selected_piece)
            # time.sleep(100)
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

    # train_dataset = train_dataset.shuffle(seed=42)
    # test_dataset = test_dataset.shuffle(seed=42)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train_only_shape_analysis_single_piece_4krect_1024resolution.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test_only_shape_analysis_single_piece_4krect_1024resolution.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
