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
from .utils.outie_innie_tool import analyze_edges_for_piece

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class JigsawPieceEdgeClassifierTool(BaseTool):
    """A tool for classifying the edges of a jigsaw puzzle piece.

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "jigsaw_piece_edge_classifier",
                "description": "Records the morphological classification (innie, outie, or flat) for each of the four edges of a specific jigsaw puzzle piece.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "piece_idx": {
                            "type": "integer",
                            "description": "The 1-based index of the puzzle piece being analyzed."
                        },
                    "required": ["piece_idx"],
                },
            }
        })
        """
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
            piece_idx = str(parameters.get("piece_idx", ""))

            gt_folder = self._instance_dict[instance_id]['ground_truth']['root_path']
            jigsaw_type = self._instance_dict[instance_id]['ground_truth']['type']
            image_path = os.path.join(gt_folder, self._instance_dict[instance_id]['ground_truth']['board_id_to_filename'][piece_idx])

            edges = analyze_edges_for_piece(image_path, jigsaw_type)
            print(instance_id, 'edges: ', edges, 'gt_folder:', gt_folder, 'piece_idx:', {piece_idx})
        except Exception as e:
            print(f'Error when executing tool: {e}, piece_idx: {piece_idx}')
            edges = None

        # reward = await self.calc_reward(instance_id)
        # # penalty for non improved answer submission
        # tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        # # update the reward
        # self._instance_dict[instance_id]["reward"] = reward

        return (
            ToolResponse(
                text=f"The morphological classification of jigsaw piece {piece_idx} is {edges}."
            ),
            1.0 if edges is not None else 0.0,
            {"success": True if edges is not None else False, "edges": edges},
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
