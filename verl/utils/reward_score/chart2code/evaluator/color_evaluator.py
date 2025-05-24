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
from itertools import permutations
from multiprocessing import Pool, cpu_count
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor   # or just串行

from scipy.optimize import linear_sum_assignment
import numpy as np

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

        prefix = self._get_prefix()
        output_file = code_file.replace(".py", "_log_colors.txt")
        suffix = self._get_suffix(output_file)
        code = prefix + code + suffix

        code_log_texts_file = code_file.replace(".py", "_log_colors.py")
        with open(code_log_texts_file, 'w') as f:
            f.write(code)
        
        os.system(f"python3 {code_log_texts_file}")

        if os.path.exists(output_file) == True:
            with open(output_file, 'r') as f:
                colors = f.read()
                colors = eval(colors)
            os.remove(output_file)
        else:
            colors = []

        os.remove(code_log_texts_file)                        
        
        # pdf_file = re.findall(r"plt\.savefig\('(.*)'\)", code)
        # if len(pdf_file) != 0:
            # pdf_file = pdf_file[0]
            # if os.path.basename(pdf_file) == pdf_file:
                # os.remove(pdf_file)

        return colors

    def _calculate_metrics(self, generation_colors: List[Tuple], golden_colors: List[Tuple]):
        try:
            generation_colors = list(generation_colors)
            golden_colors = list(golden_colors)

            group_generation_colors = group_color(generation_colors)
            group_golden_colors = group_color(golden_colors)

            # print("group_generation_colors", group_generation_colors)
            # print("group_golden_colors", group_golden_colors)
            

            # print("generation_colors", generation_colors)
            # print("golden_colors", golden_colors)

            def calculate_similarity_serial(lst1, lst2):
                if len(lst1) == 0 or len(lst2) == 0:
                    return 0

                shorter, longer = (lst1, lst2) if len(lst1) <= len(lst2) else (lst2, lst1)

                max_total_similarity = float('-inf')
                best_index = None

                for perm in permutations(longer, len(shorter)):
                    current_similarity = sum( calculate_similarity_single(c1, c2) for c1, c2 in zip(shorter, perm) )
                    current_similarity /= len(shorter)
                    
                    if current_similarity > max_total_similarity:
                        max_total_similarity = current_similarity
                        best_index = [shorter, perm]

                # best_index[0] = sorted(best_index[0])
                # best_index[1] = sorted(best_index[1])
                # print("best_index", best_index)
                for i1, i2 in zip(best_index[0], best_index[1]):
                    print(i1, i2)
                tmp_similarity = sum( calculate_similarity_single(c1, c2) for c1, c2 in zip(best_index[0], best_index[1]) ) / len(shorter)
                print("tmp_similarity", tmp_similarity)

                return max_total_similarity


            def calculate_similarity_parallel(lst1, lst2):
                if len(lst1) == 0 or len(lst2) == 0:
                    return 0

                shorter, longer = (lst1, lst2) if len(lst1) <= len(lst2) else (lst2, lst1)
                perms = permutations(longer, len(shorter))

                # create processes according to the number of CPUs
                with Pool(processes=cpu_count()) as pool:
                    similarities = pool.map(calculate_similarity_for_permutation, [(shorter, perm) for perm in perms])


                # print("length of similarities", len(similarities))

                # indexes = [item[0] for item in similarities]
                # similarities = [item[1] for item in similarities]

                # get max similarity and its index
                # max_total_similarity = max(similarities)
                # max_index = similarities.index(max_total_similarity)
                # index = indexes[max_index]

                # max_total_similarity = max(similarities)
                # index[0] = sorted(index[0])
                # index[1] = sorted(index[1])
                # for i1, i2 in zip(index[0], index[1]):
                    # print(i1, i2)

                # tmp_similarity = sum( calculate_similarity_single(c1, c2) for c1, c2 in zip(index[0], index[1]) ) / len(shorter)
                # print("tmp_similarity", tmp_similarity)
                # print("best_index", index)

                return max(similarities)

            # merge keys in group_generation_colors and group_golden_colors
            merged_color_group = list( set( list(group_generation_colors.keys()) + list(group_golden_colors.keys()) ) )
            for color in merged_color_group:
                if color not in group_generation_colors:
                    group_generation_colors[color] = []
                if color not in group_golden_colors:
                    group_golden_colors[color] = []
            
            max_set_similarity = 0

            for color in merged_color_group:
                max_set_similarity += calculate_similarity_hungarian(group_generation_colors[color], group_golden_colors[color])

            # self.metrics["similarity"] = calculate_similarity_parallel(generation_colors, golden_colors)
            # max_set_similarity = calculate_similarity_parallel(generation_colors, golden_colors)
            self.metrics["precision"] = max_set_similarity / len(generation_colors) if len(generation_colors) != 0 else 0
            if "box" in self.golden_code_file:
                self.metrics["recall"] = max_set_similarity / len(golden_colors) if len(golden_colors) != 0 else 0
            else:
                self.metrics["recall"] = max_set_similarity / len(golden_colors)
            if self.metrics["precision"] + self.metrics["recall"] == 0:
                self.metrics["f1"] = 0
            else:
                self.metrics["f1"] = 2 * self.metrics["precision"] * self.metrics["recall"] / (self.metrics["precision"] + self.metrics["recall"])

            return
        except:
            self.metrics["f1"] = None
            return

    def _get_prefix(self):
        with open(os.environ["PROJECT_PACK_PATH"]+"/evaluator/color_evaluator_prefix.py", "r") as f:
            prefix = f.read()
        return prefix
#     def _get_prefix(self):
#         return f"""
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# import sys
# sys.path.append('{os.environ['PROJECT_PACK_PATH']}')

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.axes._base import _process_plot_var_args
# from matplotlib.axes._axes import Axes
# import matplotlib.colors as mcolors
# import inspect

# drawed_colors = []

# def convert_color_to_hex(color):
#     'Convert color from name, RGBA, or hex to a hex format.'
#     try:
#         # First, try to convert from color name to RGBA to hex
#         if isinstance(color, str):
#             # Check if it's already a hex color (start with '#' and length either 7 or 9)
#             if color.startswith('#') and (len(color) == 7 or len(color) == 9):
#                 return color.upper()
#             else:
#                 return mcolors.to_hex(mcolors.to_rgba(color)).upper()
#         # Then, check if it's in RGBA format
#         elif isinstance(color, (list, tuple)) and len(color) == 4:
#             return mcolors.to_hex(color).upper()
#         else:
#             raise ValueError("Unsupported color format")
#     except ValueError as e:
#         print(color)
#         print("Error converting color:", e)
#         return None

# def log_function(func):
#     def wrapper(*args, **kwargs):
#         global drawed_colors

#         func_name = inspect.getfile(func) + "/" + func.__name__
        
#         result = func(*args, **kwargs)

#         if func.__name__ == "_makeline":
#             color = convert_color_to_hex(result[1]["color"])
#             drawed_colors.append( func_name + "--" + color )
#         elif func.__name__ == "axhline":
#             color = convert_color_to_hex(result.get_color())
#             drawed_colors.append( func_name + "--" + color )
#         elif func.__name__ == "axvline":
#             color = convert_color_to_hex(result.get_color())
#             drawed_colors.append( func_name + "--" + color )
#         elif func.__name__ == "_fill_between_x_or_y":
#             color = convert_color_to_hex(list(result.get_facecolors()[0]))
#             drawed_colors.append( func_name + "--" + color )
#         elif func.__name__ == "bar":
#             for item in result:
#                 color = convert_color_to_hex( list(item._original_facecolor))
#                 drawed_colors.append( func_name + "--" + color )
#         elif func.__name__ == "scatter":
#             # check whether cmap is used
#             if "cmap" in kwargs and kwargs["cmap"] is not None:
#                 print( "cmap is used", kwargs["cmap"] )
#                 drawed_colors.append( func_name + "--" + kwargs["cmap"] )
#             else:
#                 color = convert_color_to_hex(list(result.get_facecolor()[0]))
#                 drawed_colors.append( func_name + "--" + color )
#         elif func.__name__ == "pie":
#             for item in result[0]:
#                 color = convert_color_to_hex( item.get_facecolor() )
#                 drawed_colors.append( func_name + "--" + color )
#         elif func.__name__ == "axvspan":
#             color = convert_color_to_hex(result.get_facecolor())
#             drawed_colors.append( func_name + "--" + color )
#         elif func.__name__ == "axhspan":
#             color = convert_color_to_hex(result.get_facecolor())
#             drawed_colors.append( func_name + "--" + color )
#         return result
    
#     return wrapper

# _process_plot_var_args._makeline = log_function(_process_plot_var_args._makeline)
# Axes.bar = log_function(Axes.bar)
# Axes.scatter = log_function(Axes.scatter)
# Axes.axhline = log_function(Axes.axhline)
# Axes.axvline = log_function(Axes.axvline)
# Axes._fill_between_x_or_y = log_function(Axes._fill_between_x_or_y)
# Axes.pie = log_function(Axes.pie)
# Axes.axvspan = log_function(Axes.axvspan)
# Axes.axhspan = log_function(Axes.axhspan)
# """
    
    def _get_suffix(self, output_file):
        return f"""
drawed_colors = list(set(drawed_colors))
drawed_colors = update_drawed_colors(drawed_objects)
if len(drawed_colors) > 10:
    drawed_colors = filter_color(drawed_colors)
# print("drawed_colors", drawed_colors)
# print("len(drawed_colors)", len(drawed_colors))
# print("Length of drawed_obejcts", len(drawed_objects))
# print("drawed_objects", drawed_objects)
with open('{output_file}', 'w') as f:
    f.write(str(drawed_colors))
"""