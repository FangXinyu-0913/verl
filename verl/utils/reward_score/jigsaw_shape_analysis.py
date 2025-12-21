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
    shape_score = 0
    print(f'{solution_str}')

    # print('solution_str: ', solution_str)
    # print('ground_truth: ', ground_truth)
    # time.sleep(10)

    if '<Thinking>' in solution_str:
        format_score += 0.5

    if '<Shape Analysis>' in solution_str:
        format_score += 0.5
        try:
            shape = solution_str.split('<Shape Analysis>')[-1].split('</Shape Analysis>')[0]
            shape_json = eval(shape)
            index = 0
            shape_score = 0
            overall_score = 0
            for key, value in shape_json.items():
                gt_value_list = list(ground_truth['pieces'][index]['edges'].values())
                for val, gt_val in zip(value, gt_value_list):
                    if val == gt_val:
                        shape_score += 1
                    overall_score += 1
                index += 1
            shape_score = shape_score / overall_score
        except Exception as e:
            print(f'Error in shape analysis: {e}, org solution: {solution_str[-200:]}')
            shape_score = 0

    
    print('format_score: ', format_score)
    print('shape_score: ', shape_score)

    return format_score_percent * format_score + (1 - format_score_percent) * shape_score

def compute_score_single(solution_str, ground_truth, format_score_percent=0.00):
    """The scoring function for JigSaw.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    import time
    format_score = 0
    shape_score = 0

    # print('solution_str: ', solution_str)
    # print('ground_truth: ', ground_truth)
    # time.sleep(100)
    

    # if '<Thinking>' in solution_str:
    #     format_score += 0.5

    # if '<Shape Analysis>' in solution_str:
    #     format_score += 0.5
    try:
        # shape = solution_str.split('**Output Example:**')[-1]
        shape = list(eval(solution_str).values())
        gt = list(ground_truth['edges'].values())

        if len(gt) != len(shape):
            print('shape mismatch with gt length!')
            shape_score = 0
        else:
            for gt_item, shape_item in zip(gt, shape):
                if gt_item == shape_item:
                    shape_score += 1
            shape_score /= len(gt)
    except Exception as e:
        print(f'Error in shape analysis: {e}, org solution: {solution_str[-200:]}')
        shape_score = 0

    
    # print('format_score: ', format_score)
    print('shape_score: ', shape_score)
    # time.sleep(100)

    return shape_score