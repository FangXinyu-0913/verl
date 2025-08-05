# import re
# import os
# from dotenv import load_dotenv
# load_dotenv()
# import sys
# import numpy as np
# import concurrent.futures as cf
# sys.path.insert(0,os.environ['PROJECT_PACK_PATH'])
# from evaluator.text_evaluator import TextEvaluator
# from evaluator.chart_type_evaluator import ChartTypeEvaluator
# from evaluator.color_evaluator import ColorEvaluator
# from evaluator.layout_evaluator import LayoutEvaluator
# from evaluator.text_fast_evaluator import FastTextEvaluator
# from evaluator.layout_fast_evaluator import FastLayoutEvaluator
# from multiprocessing import get_context
# from concurrent.futures import ProcessPoolExecutor
# import tempfile,importlib



# TEXT_EVAL = TextEvaluator(use_position=False, use_axs=False)
# CHART_EVAL = ChartTypeEvaluator()
# COLOR_EVAL = ColorEvaluator()
# LAYOUT_EVAL = LayoutEvaluator()
# EVAL_CONFIG = [
#     ("evaluator.text_evaluator",   "TextEvaluator",
#         {"use_position": False, "use_axs": False}),
#     ("evaluator.chart_type_evaluator",  "ChartTypeEvaluator",   {}),
#     ("evaluator.color_evaluator",       "ColorEvaluator",       {}),
#     ("evaluator.layout_evaluator", "LayoutEvaluator",  {}),
# ]
# EVALS = [TEXT_EVAL, CHART_EVAL, COLOR_EVAL, LAYOUT_EVAL]
# _CTX   = get_context("spawn")
# _POOL  = ProcessPoolExecutor(max_workers=len(EVAL_CONFIG), mp_context=_CTX)

# def extract_python_blocks(text: str) -> list[str]:
#     """
#     从传入的 Markdown 字符串中提取所有以 ```python 开头、``` 结尾的代码块。
#     返回一个去除首尾空白的代码片段列表。
#     """
#     pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
#     code_blocks = '\n\n'.join(pattern.findall(text))

#     return code_blocks

# def _write_tmp(code: str, prefix: str) -> str:
#     "写到 /dev/shm，训练结束自动清理"
#     tmp = tempfile.NamedTemporaryFile(dir="/dev/shm", suffix=".py",
#                                       prefix=prefix, delete=False,
#                                       mode="w", encoding="utf8")
#     tmp.write(code); tmp.flush(); tmp.close()
#     return tmp.name

# def _run_eval(mod_path: str, cls_name: str, kwargs: dict,
#               gen_file: str, gold_file: str):
#     mod  = importlib.import_module(mod_path)
#     cls  = getattr(mod, cls_name)
#     ev   = cls(**kwargs)               # 实例化发生在子进程
#     ev(generation_code_file=gen_file,
#        golden_code_file=gold_file)
#     return ev.metrics.get("f1")

# def compute_score(model_output: str, ground_truth: dict) -> float:
#     """返回 4 个 evaluator 的 F1 均值（缺省视为 0）"""
#     if "```python" not in model_output:
#         return 0.0

#     gt_code  = extract_python_blocks(ground_truth["answer"])
#     out_code = extract_python_blocks(model_output)
#     plot_index = int(ground_truth['index'])

#     gold_file = f"{os.environ['PROJECT_STORE_PATH']}/{plot_index}_gt_org.py" 
#     gen_file = f"{os.environ['PROJECT_STORE_PATH']}/{plot_index}_generated_by_model_in_training.py" 
#     with open(gold_file, 'w') as f:
#         f.write(gt_code)
#     with open(gen_file, 'w') as f:
#         f.write(out_code)

#     # 提交到进程池并行执行
#     futures = [
#         _POOL.submit(_run_eval, mod, cls, kw, gen_file, gold_file)
#         for (mod, cls, kw) in EVAL_CONFIG
#     ]
#     f1s = [f.result() for f in futures if f.result() is not None]

#     # 删除临时文件
#     try: os.unlink(gen_file); os.unlink(gold_file)
#     except FileNotFoundError: pass

#     return float(np.mean(f1s)) if f1s else 0.0

        
## original init file
import re
import os
from dotenv import load_dotenv
load_dotenv()
import sys
import numpy as np
from multiprocessing import get_context
import subprocess
import concurrent.futures as cf
sys.path.insert(0,os.environ['PROJECT_PACK_PATH'])
from evaluator.text_evaluator import TextEvaluator
from evaluator.chart_type_evaluator import ChartTypeEvaluator
from evaluator.color_evaluator import ColorEvaluator
from evaluator.layout_evaluator import LayoutEvaluator

# ───────────────────────── 1) 评测器配置 ──────────────────────────
_EVAL_CFG = [
    (TextEvaluator,       dict(use_position=False, use_axs=False)),
    (ChartTypeEvaluator,  {}),
    (ColorEvaluator,      {}),
    (LayoutEvaluator,     {}),
]

# # ───────────────────────── 2) 进程池（模块级单例） ────────────────
# _CTX   = get_context("spawn")                 # 安全的启动方式
# _POOL  = cf.ProcessPoolExecutor(
#             max_workers=len(_EVAL_CFG),
#             mp_context=_CTX)

# def extract_python_blocks(text: str) -> list[str]:
#     """
#     从传入的 Markdown 字符串中提取所有以 ```python 开头、``` 结尾的代码块。
#     返回一个去除首尾空白的代码片段列表。
#     """
#     pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
#     code_blocks = '\n\n'.join(pattern.findall(text))

#     return code_blocks

# def _run_eval(cls, kwargs, gen_file, gold_file):
#     """
#     在子进程里实例化 evaluator 并返回 f1
#     """
#     ev = cls(**kwargs)                        # 锁对象只活在子进程
#     ev(generation_code_file=gen_file,
#        golden_code_file=gold_file)
#     return ev.metrics.get("f1")

# def compute_score(model_output: str, ground_truth: dict) -> float:
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

#         try:
#             # 使用 subprocess 来执行 Python 脚本并捕获错误
#             result = subprocess.run(f"python3 {generated_py_file}", shell=True, check=True, capture_output=True, text=True)
#             print(result.stdout)  # 输出正常的标准输出
#         except subprocess.CalledProcessError as e:
#             print(f"Error occurred while running the script: {e}")
#             print(f"Error output: {e.stderr}")
#             return 0.0

#         print('calculate each sub-category reward score')
#         futures = [
#             _POOL.submit(_run_eval, cls, kwargs,
#                         generated_py_file, original_py_file)
#             for cls, kwargs in _EVAL_CFG
#         ]
#         f1s = []
#         for cls, fut in zip([c for c, _ in _EVAL_CFG], futures):
#             try:
#                 score = fut.result()
#                 if score is not None:
#                     f1s.append(score)
#                     print(f"{cls.__name__} f1 = {score:.4f}")
#                 else:
#                     print(f"{cls.__name__} produced no valid f1")
#             except Exception as e:
#                 print(f"{cls.__name__} crashed: {e}")

#         return float(np.mean(f1s)) if f1s else 0.0

        # text_evaluator = TextEvaluator(use_position=False, use_axs=False)
        # chart_type_evaluator = ChartTypeEvaluator()
        # color_evaluator = ColorEvaluator()
        # layout_evaluator = LayoutEvaluator()
        # text_evaluator(
        #     generation_code_file=generated_py_file,
        #     golden_code_file=original_py_file
        # )

        # chart_type_evaluator(
        #     generation_code_file=generated_py_file,
        #     golden_code_file=original_py_file
        # )

        # color_evaluator(
        #     generation_code_file=generated_py_file,
        #     golden_code_file=original_py_file
        # )

        # layout_evaluator(
        #     generation_code_file=generated_py_file,
        #     golden_code_file=original_py_file
        # )

        # f1_score = []
        # evaluator_list = [text_evaluator, chart_type_evaluator, color_evaluator, layout_evaluator] 
        # for evaluator in evaluator_list:
        #     if evaluator.metrics['f1'] != None:
        #         f1_score.append(evaluator.metrics['f1'])
        #         print(f"{evaluator} has valid f1 score as reward: {evaluator.metrics['f1']}")
        #     else:
        #         print(f"{evaluator} has no valid f1 score as reward")
        
        # return np.mean(f1_score)




from pathlib import Path
import numpy as np, concurrent.futures as cf

# --- A) helper: in-proc exec -------------------------------------------------
def _safe_exec(path: str) -> bool:
    import importlib.util, contextlib, io, gc, matplotlib.pyplot as plt
    try:
        spec = importlib.util.spec_from_loader(path, loader=None)
        mod  = importlib.util.module_from_spec(spec)
        src  = Path(path).read_text(encoding="utf-8")
        buf  = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(src, path, "exec"), mod.__dict__)
        if buf.tell():
            print(buf.getvalue())
        plt.close("all"); gc.collect()
        return True
    except Exception as e:
        print(f"[exec] {path} crashed: {e}")
        return False

# --- B) lightweight evaluator run -------------------------------------------
def _run_eval_direct(cls, kwargs, gen, gold):
    ev = cls(**kwargs)
    ev(generation_code_file=gen, golden_code_file=gold)
    return ev.metrics.get("f1")

# --- D) regex pre-compile ----------------------------------------------------
_PY_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
def extract_python_blocks(md: str) -> str:
    return "\n\n".join(_PY_RE.findall(md))

# ---------------------------------------------------------------------------

def compute_score(model_output: str, ground_truth: dict) -> float:
    if "```python" not in model_output:
        return 0.0

    gt_code      = extract_python_blocks(ground_truth["answer"])
    output_code  = extract_python_blocks(model_output)
    idx          = int(ground_truth["index"])

    store_path   = os.environ["PROJECT_STORE_PATH"]
    original_py  = f"{store_path}/{idx}_gt_org.py"
    generated_py = f"{store_path}/{idx}_generated_by_model_in_training.py"

    Path(original_py).write_text(gt_code,      encoding="utf-8")
    Path(generated_py).write_text(output_code, encoding="utf-8")

    # -------- A) run generated script ---------------------------------------
    if not _safe_exec(generated_py):
        return 0.0     # 脚本报错，直接给 0 分

    # -------- B) evaluator -----------------------------------------------
    with cf.ThreadPoolExecutor(max_workers=len(_EVAL_CFG)) as tp:
        futures = [
            tp.submit(_run_eval_direct, cls, kwargs, generated_py, original_py)
            for cls, kwargs in _EVAL_CFG
        ]

    f1s = []
    for (cls, _), fut in zip(_EVAL_CFG, futures):
        try:
            v = fut.result()
            if v is not None:
                f1s.append(v)
                print(f"{cls.__name__} f1 = {v:.4f}")
            else:
                print(f"{cls.__name__}: no valid f1")
        except Exception as e:
            print(f"{cls.__name__} crashed: {e}")

    return float(np.mean(f1s)) if f1s else 0.0