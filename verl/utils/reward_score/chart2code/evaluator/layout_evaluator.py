from typing import List, Tuple, Any
from dotenv import load_dotenv
load_dotenv()

import os
import sys,pathlib
from pathlib import Path
sys.path.append(os.environ["PROJECT_PACK_PATH"])

import matplotlib.pyplot as plt
import eval_configs.global_config as gloabl_config
from concurrent.futures import ThreadPoolExecutor   # or just串行

class LayoutEvaluator:

    def __init__(self) -> None:
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "f1": 0
        }

    def __call__(self, generation_code_file, golden_code_file):
        with ThreadPoolExecutor(max_workers=2) as tp:
            fut_gen  = tp.submit(self._log_layouts, generation_code_file)
            fut_gold = tp.submit(self._log_layouts, golden_code_file)
            generation_layouts, golden_layouts = fut_gen.result(), fut_gold.result()
        
        self._calculate_metrics(generation_layouts, golden_layouts)

        redundant_pdf = (
            Path(os.environ["PROJECT_STORE_PATH"])
            / Path(golden_code_file).with_suffix(".pdf").name
        )
        if redundant_pdf.exists():
            redundant_pdf.unlink()

        # print(self.metrics)


    def _log_layouts(self, code_file):
        """
        Get objects of the code
        """

        with open(code_file, 'r') as f:
            lines = f.readlines()
        code = ''.join(lines)

        prefix = self._get_prefix()
        output_file = code_file.replace(".py", "_log_layouts.txt")
        if "/graph" in code_file:
            suffix = self._get_suffix_special_for_graph(output_file)
        else:
            suffix = self._get_suffix(output_file)

        code = prefix + code + suffix

        code_log_texts_file = code_file.replace(".py", "_log_layouts.py")
        with open(code_log_texts_file, 'w') as f:
            f.write(code)
        
        os.system(f"python3 {code_log_texts_file}")

        if os.path.exists(output_file) == True:                
            with open(output_file, 'r') as f:
                texts = f.read()
                texts = eval(texts)
            os.remove(output_file)
        else:
            texts = []
        os.remove(code_log_texts_file)

        return texts

    def _calculate_metrics(self, generation_layouts, golden_layouts):
        if not generation_layouts or not golden_layouts:
            self.metrics = dict.fromkeys(self.metrics, 0.0); return

        # 把 dict → tuple(sorted(...)) 后再做集合交集
        freeze = lambda d: tuple(sorted(d.items()))
        gen_set  = set(map(freeze, generation_layouts))
        gold_set = set(map(freeze, golden_layouts))

        n_correct = len(gen_set & gold_set)
        prec = n_correct / len(gen_set)
        rec  = n_correct / len(gold_set)
        f1   = 0.0 if not (prec or rec) else 2*prec*rec/(prec+rec)
        self.metrics.update(precision=prec, recall=rec, f1=f1)

    def _get_prefix(self):
        return f"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
sys.path.append('{os.environ['PROJECT_PACK_PATH']}')
"""
    
    def _get_suffix(self, output_file):
        return f"""

def get_gridspec_layout_info(fig):
    layout_info = {{}}
    for ax in fig.axes:
        spec = ax.get_subplotspec()
        if spec is None:
            continue
        gs = spec.get_gridspec()
        nrows, ncols = gs.get_geometry()
        row_start, row_end = spec.rowspan.start, spec.rowspan.stop - 1  # Zero-based and inclusive
        col_start, col_end = spec.colspan.start, spec.colspan.stop - 1  # Zero-based and inclusive
        layout_info[ax] = dict(nrows=nrows, ncols=ncols, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)
    # print(layout_info)
    layout_info = list(layout_info.values())
    return layout_info

layout_info = get_gridspec_layout_info(fig=plt.gcf())
with open('{output_file}', 'w') as f:
    f.write(str(layout_info))
"""
    
    def _get_suffix_special_for_graph(self, output_file):
        return f"""
def get_gridspec_layout_info(fig):
    layout_info = {{}}
    for ax in fig.axes:
        layout_info[ax] = dict(nrows=1, ncols=1, row_start=0, row_end=1, col_start=0, col_end=1)
    # print(layout_info)
    layout_info = list(layout_info.values())
    return layout_info

layout_info = get_gridspec_layout_info(fig=plt.gcf())
with open('{output_file}', 'w') as f:
    f.write(str(layout_info))
"""