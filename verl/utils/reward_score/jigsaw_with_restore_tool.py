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

# _SOLUTION_CLIP_CHARS = 300


# def extract_solution(solution_str, method="strict"):
#     assert method in ["strict", "flexible"]

#     # Optimization: Regular expression matching on very long strings can be slow.
#     # For math problems, the final answer is usually at the end.
#     # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
#     if len(solution_str) > _SOLUTION_CLIP_CHARS:
#         solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

#     if method == "strict":
#         # this also tests the formatting of the model
#         solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
#         if len(solutions) == 0:
#             final_answer = None
#         else:
#             # take the last solution
#             final_answer = solutions[-1].replace(",", "").replace("$", "")
#     elif method == "flexible":
#         answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
#         final_answer = None
#         if len(answer) == 0:
#             # no reward is there is no answer
#             pass
#         else:
#             invalid_str = ["", "."]
#             # find the last number that is not '.'
#             for final_answer in reversed(answer):
#                 if final_answer not in invalid_str:
#                     break
#     return final_answer


# def compute_score(solution_str, ground_truth, extra_info):
#     """The scoring function for JigSaw.

#     Args:
#         solution_str: the solution text
#         ground_truth: the ground truth
#         method: the method to extract the solution, choices are 'strict' and 'flexible'
#         format_score: the score for the format
#         score: the score for the correct answer
#     """
#     format_score = 0
#     final_answer_score = 0
#     acc_score_percent = 1.5
#     format_score_percent = 0.2
#     tool_using_percent = 0.5

#     import time
#     # print('extra_info: ', extra_info)
#     # print('solution_str: ', solution_str)
#     # print('ground_truth: ', ground_truth)
#     # time.sleep(1000)

#     if '<Thinking>' in solution_str:
#         format_score += 0.5

#     if '<Final Answer>' in solution_str:
#         format_score += 0.5
#         try:
#             raw_answer = solution_str.split('<Final Answer>')[-1].split('</Final Answer>')[0]
#             print(raw_answer)
#             if '(' in raw_answer:
#                 cleaned = raw_answer.replace('(', '').replace(')', '').split(',')
#                 coords = [int(x.strip()) for x in cleaned if x.strip()]
                
#                 final_answer = []
#                 # 每次取两个数 (row, col)
#                 for i in range(0, len(coords), 2):
#                     if i + 1 < len(coords):
#                         r, c = coords[i], coords[i+1]
#                         # (1,1)->1, (1,2)->2, (2,1)->3, (2,2)->4
#                         linear_idx = (r - 1) * 2 + c
#                         final_answer.append(linear_idx)
#             elif '[' in raw_answer:
#                 final_answer = eval(raw_answer)
#             else:
#                 final_answer = [int(x.strip()) for x in raw_answer.split(',') if x.strip()]

#             gt_final_answer = ground_truth['solution']
#             if len(final_answer) == len(gt_final_answer):
#                 final_answer_score = sum(final_answer[i] == gt_final_answer[i] for i in range(len(final_answer))) / len(final_answer)
#             else:
#                 final_answer_score = 0
#         except Exception as e:
#             print(f'Error in final answer: {e}, org solution: {solution_str[:2000]}')
#             final_answer_score = 0

#     print('tool_rewards: ', extra_info['tool_rewards'])
#     print('output_index: ', extra_info['output_index'])

#     tool_score = 0
#     if sum(extra_info['tool_rewards']) > 0:
#         tool_score = 1.0 * len(list(set(extra_info['output_index']))) / 16 # if sum(extra_info['tool_rewards']) < 4 else 1.0
#         if final_answer_score > 0:
#             tool_score += 0.5
#         # elif final_answer_score > 0:
#         #     if sum(extra_info['tool_rewards']) >= 2:
#         #         tool_score = 0.6
#         #     else:
#         #         tool_score = 0.3 if * sum(extra_info['tool_rewards']) 

#     # tool_score = 1.0 if len(extra_info['tool_rewards']) > 0 and sum(extra_info['tool_rewards']) > 0 and final_answer_score >= 0.375 else 0
#     # print('format_score: ', format_score, format_score_percent)
#     # print('shape_score: ', shape_score)
#     # print('final_answer_score: ', final_answer_score, (1 - format_score_percent))
#     # print('final score: ', format_score_percent * format_score + (1 - format_score_percent) * final_answer_score)
#     print('tool_score: ', tool_score)
#     print('final_answer_score: ', final_answer_score)
#     print('format_score: ', format_score)
#     final_score = acc_score_percent * final_answer_score + tool_using_percent * tool_score + format_score_percent * (format_score - 1)
#     print('final_score: ', final_score)
    
#     return final_score

    # answer = extract_solution(solution_str=solution_str, method=method)
    # if answer is None:
    #     return 0
    # else:
    #     if answer == ground_truth:
    #         return score
    #     else:
    #         return format_score

def compute_score(solution_str, ground_truth, extra_info):
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
    acc_score_percent = 1.2
    format_score_percent = 0.2
    tool_using_percent = 0.8

    import time

    if '<Thinking>' in solution_str:
        format_score += 0.5

    if '<Final Answer>' in solution_str:
        format_score += 0.5
        try:
            raw_answer = solution_str.split('<Final Answer>')[-1].split('</Final Answer>')[0]
            print('raw_answer in restore_tool_reward_function', raw_answer)
            if '(' in raw_answer:
                cleaned = raw_answer.replace('(', '').replace(')', '').split(',')
                coords = [int(x.strip()) for x in cleaned if x.strip()]
                
                final_answer = []
                # 每次取两个数 (row, col)
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        r, c = coords[i], coords[i+1]
                        # (1,1)->1, (1,2)->2, (2,1)->3, (2,2)->4
                        linear_idx = (r - 1) * 2 + c
                        final_answer.append(linear_idx)
            elif '[' in raw_answer:
                final_answer = eval(raw_answer)
            else:
                final_answer = [int(x.strip()) for x in raw_answer.split(',') if x.strip()]

            gt_final_answer = ground_truth['solution']
            if len(final_answer) == len(gt_final_answer):
                final_answer_score = sum(final_answer[i] == gt_final_answer[i] for i in range(len(final_answer))) / len(final_answer)
            else:
                final_answer_score = 0
        except Exception as e:
            print(f'Error in final answer: {e}, org solution: {solution_str[:2000]}')
            final_answer_score = 0

    print('tool_rewards: ', extra_info['tool_rewards'])

    tool_score = 0
    if sum(extra_info['tool_rewards']) > 0:
        if final_answer_score > 0:
            tool_score = 0.2
        if final_answer_score > 0.2:
            tool_score = 0.5
        if final_answer_score > 0.5:
            tool_score = 0.8
        if final_answer_score > 0.8:
            tool_score = 1


    print('tool_score: ', tool_score)
    print('final_answer_score: ', final_answer_score)
    print('format_score: ', format_score)
    final_score = acc_score_percent * final_answer_score + tool_using_percent * tool_score + format_score_percent * (format_score - 1)
    print('final_score: ', final_score)
    
    return final_score