import re
import os
from dotenv import load_dotenv
load_dotenv()
import sys
import numpy as np
import concurrent.futures as cf
sys.path.insert(0,os.environ['PROJECT_PACK_PATH'])
from evaluator.text_evaluator import TextEvaluator
from evaluator.chart_type_evaluator import ChartTypeEvaluator
from evaluator.color_evaluator import ColorEvaluator
from evaluator.layout_evaluator import LayoutEvaluator
from evaluator.text_fast_evaluator import FastTextEvaluator
from evaluator.layout_fast_evaluator import FastLayoutEvaluator
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
import tempfile,importlib



TEXT_EVAL = TextEvaluator(use_position=False, use_axs=False)
CHART_EVAL = ChartTypeEvaluator()
COLOR_EVAL = ColorEvaluator()
LAYOUT_EVAL = LayoutEvaluator()
EVAL_CONFIG = [
    ("evaluator.text_evaluator",   "TextEvaluator",
        {"use_position": False, "use_axs": False}),
    ("evaluator.chart_type_evaluator",  "ChartTypeEvaluator",   {}),
    ("evaluator.color_evaluator",       "ColorEvaluator",       {}),
    ("evaluator.layout_evaluator", "LayoutEvaluator",  {}),
]
EVALS = [TEXT_EVAL, CHART_EVAL, COLOR_EVAL, LAYOUT_EVAL]
_CTX   = get_context("spawn")
_POOL  = ProcessPoolExecutor(max_workers=len(EVAL_CONFIG), mp_context=_CTX)

def extract_python_blocks(text: str) -> list[str]:
    """
    从传入的 Markdown 字符串中提取所有以 ```python 开头、``` 结尾的代码块。
    返回一个去除首尾空白的代码片段列表。
    """
    pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    code_blocks = '\n\n'.join(pattern.findall(text))

    return code_blocks

def _write_tmp(code: str, prefix: str) -> str:
    "写到 /dev/shm，训练结束自动清理"
    tmp = tempfile.NamedTemporaryFile(dir="/dev/shm", suffix=".py",
                                      prefix=prefix, delete=False,
                                      mode="w", encoding="utf8")
    tmp.write(code); tmp.flush(); tmp.close()
    return tmp.name

def _run_eval(mod_path: str, cls_name: str, kwargs: dict,
              gen_file: str, gold_file: str):
    mod  = importlib.import_module(mod_path)
    cls  = getattr(mod, cls_name)
    ev   = cls(**kwargs)               # 实例化发生在子进程
    ev(generation_code_file=gen_file,
       golden_code_file=gold_file)
    return ev.metrics.get("f1")

def compute_score(model_output: str, ground_truth: dict) -> float:
    """返回 4 个 evaluator 的 F1 均值（缺省视为 0）"""
    if "```python" not in model_output:
        return 0.0

    gt_code  = extract_python_blocks(ground_truth["answer"])
    out_code = extract_python_blocks(model_output)
    plot_index = int(ground_truth['index'])

    gold_file = f"{os.environ['PROJECT_STORE_PATH']}/{plot_index}_gt_org.py" 
    gen_file = f"{os.environ['PROJECT_STORE_PATH']}/{plot_index}_generated_by_model_in_training.py" 
    with open(gold_file, 'w') as f:
        f.write(gt_code)
    with open(gen_file, 'w') as f:
        f.write(out_code)

    # 提交到进程池并行执行
    futures = [
        _POOL.submit(_run_eval, mod, cls, kw, gen_file, gold_file)
        for (mod, cls, kw) in EVAL_CONFIG
    ]
    f1s = [f.result() for f in futures if f.result() is not None]

    # 删除临时文件
    try: os.unlink(gen_file); os.unlink(gold_file)
    except FileNotFoundError: pass

    return float(np.mean(f1s)) if f1s else 0.0

        
## original init file
# import re
# import os
# from dotenv import load_dotenv
# load_dotenv()
# import sys
# import numpy as np
# sys.path.insert(0,os.environ['PROJECT_PACK_PATH'])
# from evaluator.text_evaluator import TextEvaluator
# from evaluator.chart_type_evaluator import ChartTypeEvaluator
# from evaluator.color_evaluator import ColorEvaluator
# from evaluator.layout_evaluator import LayoutEvaluator

# def extract_python_blocks(text: str) -> list[str]:
#     """
#     从传入的 Markdown 字符串中提取所有以 ```python 开头、``` 结尾的代码块。
#     返回一个去除首尾空白的代码片段列表。
#     """
#     pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
#     code_blocks = '\n\n'.join(pattern.findall(text))

#     return code_blocks



# def compute_score(model_output: str, ground_truth: dict) -> bool:
#     if "```python" not in model_output:  # no code in model output
#         return 0.0
#     else:
#         gt_code = extract_python_blocks(ground_truth['answer'])
#         output_code = extract_python_blocks(model_output)
#         plot_index = int(ground_truth['index'])

#         original_py_file = f"{os.environ['PROJECT_STORE_PATH']}/{plot_index}_gt_org.py" 
#         generated_py_file = f"{os.environ['PROJECT_STORE_PATH']}/{plot_index}_generated_by_model_in_training.py" 
#         with open(original_py_file, 'w') as f:
#             f.write(gt_code)
#         with open(generated_py_file, 'w') as f:
#             f.write(output_code)

#         text_evaluator = TextEvaluator(use_position=False, use_axs=False)
#         chart_type_evaluator = ChartTypeEvaluator()
#         color_evaluator = ColorEvaluator()
#         layout_evaluator = LayoutEvaluator()
#         text_evaluator(
#             generation_code_file=generated_py_file,
#             golden_code_file=original_py_file
#         )

#         chart_type_evaluator(
#             generation_code_file=generated_py_file,
#             golden_code_file=original_py_file
#         )

#         color_evaluator(
#             generation_code_file=generated_py_file,
#             golden_code_file=original_py_file
#         )

#         layout_evaluator(
#             generation_code_file=generated_py_file,
#             golden_code_file=original_py_file
#         )

#         f1_score = []
#         evaluator_list = [text_evaluator, chart_type_evaluator, color_evaluator, layout_evaluator] 
#         for evaluator in evaluator_list:
#             if evaluator.metrics['f1'] != None:
#                 f1_score.append(evaluator.metrics['f1'])
#                 print(f"{evaluator} has valid f1 score as reward: {evaluator.metrics['f1']}")
#             else:
#                 print(f"{evaluator} has no valid f1 score as reward")
        
#         return np.mean(f1_score)