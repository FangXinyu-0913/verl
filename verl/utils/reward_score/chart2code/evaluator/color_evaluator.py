from typing import List, Tuple, Any
from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.append(os.environ["PROJECT_PACK_PATH"])

import matplotlib.pyplot as plt
import eval_configs.global_config as gloabl_config

import re,pathlib
from pathlib import Path

from skimage.color import deltaE_cie76
from skimage.color import rgb2lab
import numpy as np
# Monkey patch numpy to add asscalar if it's missing
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item()
from itertools import permutations
from multiprocessing import Pool, cpu_count
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor   # or just串行

from scipy.optimize import linear_sum_assignment
import numpy as np

import subprocess
import time
import os
import signal
import psutil


# 注意：这个函数应该放在你的其他辅助函数旁边
def calculate_similarity_hungarian_gemini(lst1: List[str], lst2: List[str]) -> float:
    """
    使用匈牙利算法计算两组颜色之间的最大总相似度。
    """
    if not lst1 or not lst2:
        return 0.0

    # 确保 lst1 是较短或等长的列表
    if len(lst1) > len(lst2):
        lst1, lst2 = lst2, lst1

    # 创建成本矩阵。linear_sum_assignment 求的是最小成本，
    # 所以我们的成本是 (1 - 相似度)。
    cost_matrix = np.zeros((len(lst1), len(lst2)))
    for i, c1 in enumerate(lst1):
        for j, c2 in enumerate(lst2):
            similarity = calculate_similarity_single(c1, c2)
            cost_matrix[i, j] = 1 - similarity

    # 使用匈牙利算法找到成本最低的匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 从成本计算回总相似度
    # cost_matrix[row_ind, col_ind] 是最优匹配下的成本列表
    # (1 - cost) 就是相似度
    max_total_similarity = (1 - cost_matrix[row_ind, col_ind]).sum()
    
    return max_total_similarity

def _similarity_matrix(shorter, longer):
    # 计算 |shorter|×|longer| 的相似度矩阵，并取 (1-sim) 作为 cost
    mat = np.zeros((len(shorter), len(longer)))
    for i, c1 in enumerate(shorter):
        for j, c2 in enumerate(longer):
            mat[i, j] = 1 - calculate_similarity_single(c1, c2)
    return mat

def calculate_similarity_hungarian(lst1, lst2):
    if not lst1 or not lst2:
        return 0.0
    shorter, longer = (lst1, lst2) if len(lst1) <= len(lst2) else (lst2, lst1)
    cost = _similarity_matrix(shorter, longer)
    row_ind, col_ind = linear_sum_assignment(cost)
    total = (1 - cost[row_ind, col_ind]).sum()         # 把 cost 再映射回相似度
    return total / len(shorter)

def group_color(color_list):
    color_dict = {}

    for color in color_list:
        chart_type = color.split("--")[0]
        color = color.split("--")[1]

        if chart_type not in color_dict:
            color_dict[chart_type] = [color]
        else:
            color_dict[chart_type].append(color)

    return color_dict

from functools import lru_cache

@lru_cache(maxsize=None)
def _hex_to_lab(hex_color: str):
    rgb = hex_to_rgb(hex_color)
    return rgb_to_lab(rgb)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(rgb):
    """
    Convert an RGB color to Lab color space.
    RGB values should be in the range [0, 255].
    """
    # Create an sRGBColor object from RGB values
    rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)
    
    # Convert to Lab color space
    lab_color = convert_color(rgb_color, LabColor)
    
    return lab_color   

def calculate_similarity_single(c1, c2):
    if c1.startswith("#") and c2.startswith("#"):
        lab1 = _hex_to_lab(c1)
        lab2 = _hex_to_lab(c2)
        return max(0, 1 - delta_e_cie2000(lab1, lab2) / 100)
    return 1.0 if c1 == c2 else 0.0

def calculate_similarity_for_permutation(args):
    shorter, perm = args
    current_similarity = sum(calculate_similarity_single(c1, c2) for c1, c2 in zip(shorter, perm))
    return current_similarity

class ColorEvaluator:

    def __init__(self) -> None:
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

    def __call__(self, generation_code_file, golden_code_file):

        self.golden_code_file = golden_code_file

        with ThreadPoolExecutor(max_workers=2) as tp:   # GIL 影响可以忽略
            gen_fut  = tp.submit(self._log_colors, generation_code_file)
            gold_fut = tp.submit(self._log_colors, golden_code_file)
            generation_colors = gen_fut.result()
            golden_colors     = gold_fut.result()

        
        self._calculate_metrics(generation_colors, golden_colors)

        redundant_pdf = (
            Path(os.environ["PROJECT_STORE_PATH"])
            / Path(golden_code_file).with_suffix(".pdf").name
        )
        if redundant_pdf.exists():
            redundant_pdf.unlink()



    def _log_colors(self, code_file):
        """
        Get text objects of the code
        """

        with open(code_file, 'r') as f:
            lines = f.readlines()
        code = ''.join(lines)

        # 清理原脚本中的保存/显示/关闭，避免重复 I/O 放大（新加入，不知道有没有用）
        try:
            code = re.sub(r"plt\\.savefig\(.*?\)\s*", "", code, flags=re.S)
            code = re.sub(r"fig\\.savefig\(.*?\)\s*", "", code, flags=re.S)
            code = re.sub(r"plt\\.show\(.*?\)\s*", "", code, flags=re.S)
            code = re.sub(r"plt\\.close\(.*?\)\s*", "", code, flags=re.S)
            code = re.sub(r"plt\\.clf\(.*?\)\s*", "", code, flags=re.S)
        except Exception:
            pass

        prefix = self._get_prefix()
        output_file = code_file.replace(".py", "_log_colors.txt")
        suffix = self._get_suffix(output_file)
        code = prefix + code + suffix

        code_log_texts_file = code_file.replace(".py", "_log_colors.py")
        # 若已有输出，直接读取，避免重复执行
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    colors = f.read()
                    colors = eval(colors)
                os.remove(output_file)
                return colors
            except Exception:
                pass

        with open(code_log_texts_file, 'w') as f:
            f.write(code)

        # os.system(f"python3 {code_log_texts_file}")
        from .utils import execute_python_with_timeout

        execute_python_with_timeout(code_log_texts_file)


        if os.path.exists(output_file) == True:
            with open(output_file, 'r') as f:
                colors = f.read()
                colors = eval(colors)
            os.remove(output_file)
        else:
            colors = []

        # os.remove(code_log_texts_file)                        
        
        # pdf_file = re.findall(r"plt\.savefig\('(.*)'\)", code)
        # if len(pdf_file) != 0:
            # pdf_file = pdf_file[0]
            # if os.path.basename(pdf_file) == pdf_file:
                # os.remove(pdf_file)

        return colors

    def _calculate_metrics(self, generation_colors: List[Tuple], golden_colors: List[Tuple]):
        try:
            group_generation_colors = group_color(generation_colors)
            group_golden_colors = group_color(golden_colors)

            def calculate_similarity_parallel(lst1, lst2):
                if len(lst1) == 0 or len(lst2) == 0:
                    return 0

                shorter, longer = (lst1, lst2) if len(lst1) <= len(lst2) else (lst2, lst1)
                perms = permutations(longer, len(shorter))

                # create processes according to the number of CPUs
                with Pool(processes=2) as pool:
                    similarities = pool.map(calculate_similarity_for_permutation, [(shorter, perm) for perm in perms])

                return max(similarities)

            # merge keys in group_generation_colors and group_golden_colors
            # merged_color_group = list( set( list(group_generation_colors.keys()) + list(group_golden_colors.keys()) ) )
            merged_color_group = set(group_generation_colors.keys()) | set(group_golden_colors.keys())

            
            total_max_similarity = 0.0

            for group_key in merged_color_group:
                gen_list = group_generation_colors.get(group_key, [])
                gold_list = group_golden_colors.get(group_key, [])
                
                # 使用新的、高效的匈牙利算法函数
                total_max_similarity += calculate_similarity_hungarian(gen_list, gold_list)

            # 使用计算出的总相似度来计算 precision 和 recall
            self.metrics["precision"] = total_max_similarity / len(generation_colors) if generation_colors else 0.0
            self.metrics["recall"] = total_max_similarity / len(golden_colors) if golden_colors else 0.0
            
            # F1 score 计算保持不变
            if self.metrics["precision"] + self.metrics["recall"] == 0:
                self.metrics["f1"] = 0.0
            else:
                self.metrics["f1"] = 2 * self.metrics["precision"] * self.metrics["recall"] / (self.metrics["precision"] + self.metrics["recall"])
            return
        except Exception as e:
            print(f"Error in calculating color metrics: {e}")
            self.metrics["f1"] = None
            return

    def _get_prefix(self):
        with open(os.environ["PROJECT_PACK_PATH"]+"/evaluator/color_evaluator_prefix.py", "r") as f:
            prefix = f.read()
        return prefix

    
    def _get_suffix(self, output_file):
        return f"""
drawed_colors = list(set(drawed_colors))
if len(drawed_colors) > 10:
    drawed_colors = filter_color_optimized(drawed_colors)
# print("drawed_colors", drawed_colors)
# print("len(drawed_colors)", len(drawed_colors))
# print("Length of drawed_obejcts", len(drawed_objects))
# print("drawed_objects", drawed_objects)
with open('{output_file}', 'w') as f:
    f.write(str(drawed_colors))
"""