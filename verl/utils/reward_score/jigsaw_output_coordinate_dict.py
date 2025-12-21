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

import re
import json

def compute_score(solution_str, ground_truth, format_score_percent=0.1):
    """The scoring function for JigSaw.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    format_score = 0
    final_answer_score = 0

    import time
    # print('solution_str: ', solution_str)
    # print('ground_truth: ', ground_truth)
    # time.sleep(1000)

    if '<Thinking>' in solution_str:
        format_score += 0.5

    if '<Final Answer>' in solution_str:
        format_score += 0.5
        try:
            raw_answer = solution_str.split('<Final Answer>')[-1].split('</Final Answer>')[0]
            raw_answer = raw_answer.replace('{{', '{').replace('}}', '}')
            coordinate_dict = eval(raw_answer)
            gt_final_answer = ground_truth['solution']
            final_answer_score = []
            if type(gt_final_answer) == list and len(gt_final_answer) == len(coordinate_dict):
                for key, value in coordinate_dict.items():
                    if gt_final_answer[(value[0] - 1) * 2 + value[1] - 1] == int(key):
                        final_answer_score.append(1)
                    else:
                        final_answer_score.append(0)
            else:
                final_answer_score = []
        except Exception as e:
            print(f'Error in final answer: {e}, org solution: {solution_str[:2000]}')
            final_answer_score = []
        final_answer_score = sum(final_answer_score) / len(final_answer_score) if len(final_answer_score) > 0 else 0

    
    # print('format_score: ', format_score, format_score_percent)
    # print('shape_score: ', shape_score)
    # print('final_answer_score: ', final_answer_score, (1 - format_score_percent))
    print('final score: ', format_score_percent * format_score + (1 - format_score_percent) * final_answer_score)

    return format_score_percent * format_score + (1 - format_score_percent) * final_answer_score

    
