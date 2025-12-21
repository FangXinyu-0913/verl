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

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.utils.reward_score import gsm8k
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from .utils.jigsaw_solver import reassemble_jigsaw

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def has_duplicate_ignore_zero(nums):
    seen = set()
    for num in nums:
        # 如果是 0，直接跳过
        if num == 0:
            continue
        
        # 如果之前已经在集合里见过这个非0数字，说明重复了
        if num in seen:
            return True
            
        # 没见过，加入集合，继续往后看
        seen.add(num)
        
    # 循环结束都没返回 True，说明没有重复
    return False


class JigsawRestorationTool(BaseTool):
    """A tool for restoring the jigsaw puzzle.

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):

        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", None)
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            order = parameters.get("order", "")
            if not isinstance(order, list):
                order = list(order)

            self._instance_dict[instance_id]["response"] = order

            final_answer = order

            # final_answer = []
            # # 每次取两个数 (row, col)
            # for item in order:
            #     r, c = item
            #     # (1,1)->1, (1,2)->2, (2,1)->3, (2,2)->4
            #     linear_idx = (r - 1) * 2 + c
            #     final_answer.append(linear_idx)

            gt_folder = self._instance_dict[instance_id]['ground_truth']['root_path']
            target_resolution = (768, 768)
            print(instance_id, 'reconstructed answer in tool: ', final_answer)
            # final answer里面没有除了1-4外的其他数字
            if len(final_answer) == 16 and all(0 <= x <= 16 for x in final_answer) and not has_duplicate_ignore_zero(final_answer):
                canvas = reassemble_jigsaw(gt_folder, final_answer, 'rect', target_resolution)
            else:
                canvas = None
        except Exception as e:
            print(f'Error when executing tool: {e}, order: {order}')
            canvas = None

        if canvas is None:
            return (
                ToolResponse(
                    image=[None],
                    text=f"The puzzle is failed and cannot be reconstructed according to the given answers={order}. Please observe carefully and reconsider.",
                ),
                0.0,
                {"success": False, "order": order, "piece_size": [target_resolution[0], target_resolution[1]]},
            )
        else:
            return (
                ToolResponse(
                    image=[canvas],
                    # text=f"拼图拼接完成，order={order} (行优先: 左上/右上/左下/右下)。",
                    text=f"The puzzle has been completed according to the given answers={order}, and the result is shown in the image. Please observe carefully and reconsider.",
                ),
                1.0,
                {"success": True, "order": order, "piece_size": [target_resolution[0], target_resolution[1]]},
            )

    # async def calc_reward(self, instance_id: str, **kwargs) -> float:
    #     return gsm8k.compute_score(
    #         self._instance_dict[instance_id]["response"],
    #         self._instance_dict[instance_id]["ground_truth"],
    #         method="flexible",
    #         format_score=0.0,
    #         score=1.0,
    #     )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
