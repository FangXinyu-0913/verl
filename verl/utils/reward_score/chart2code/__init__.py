import os
import subprocess
import sys
sys.path.insert(0,os.environ['PROJECT_PACK_PATH'])
from evaluator.text_evaluator import TextEvaluator
from evaluator.chart_type_evaluator import ChartTypeEvaluator
from evaluator.color_evaluator import ColorEvaluator
from evaluator.layout_evaluator import LayoutEvaluator
from evaluator.text_advanced_evaluator import TextAttributeEvaluator
from evaluator.data_evaluator import DataEvaluator
from evaluator.style_evaluator import StyleAndAxesEvaluator
from evaluator.utils import execute_python_with_timeout

import re
import os
import time, uuid
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
import importlib
import numpy as np
import concurrent.futures
import functools

def with_timeout(timeout):
    """
    装饰器：给函数加上超时限制
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"{func.__name__} 超过 {timeout} 秒未完成，已超时！")
        return wrapper
    return decorator
_MPF_PLOT_PATTERN = re.compile(r"mpf\.plot\((.*?)\s*savefig=dict\(.*?\)(.*)\)", re.DOTALL)

# # TEXT_EVAL = TextEvaluator(use_position=False, use_axs=False)
# CHART_EVAL = ChartTypeEvaluator()
# COLOR_EVAL = ColorEvaluator()
# LAYOUT_EVAL = LayoutEvaluator()
# TEXT_ATTRIBUTE_EVAL = TextAttributeEvaluator(use_position=False, use_attributes=False, use_axs=True)
# DATA_EVAL = DataEvaluator()
# STYLE_AND_AXES_EVAL = StyleAndAxesEvaluator()
# EVAL_CONFIG = [
#     ("evaluator.text_advanced_evaluator",   "TextAttributeEvaluator",
#         {"use_position": False, "use_attributes": False, "use_axs": True}),
#     ("evaluator.chart_type_evaluator",  "ChartTypeEvaluator",   {}),
#     ("evaluator.color_evaluator",       "ColorEvaluator",       {}),
#     ("evaluator.layout_evaluator", "LayoutEvaluator",  {}),
#     ("evaluator.data_evaluator", "DataEvaluator",  {}),
#     ("evaluator.style_evaluator", "StyleAndAxesEvaluator",  {}),
# ]
# EVALS = [TEXT_ATTRIBUTE_EVAL, CHART_EVAL, COLOR_EVAL, LAYOUT_EVAL, DATA_EVAL, STYLE_AND_AXES_EVAL]

# def _init_eval_worker():
#     """子进程启动时预加载 Evaluator，避免每次任务重复构造"""
#     global EVAL_OBJS
#     EVAL_OBJS = []
#     for mod, cls, kw in EVAL_CONFIG:
#         mod_obj = importlib.import_module(mod)
#         cls_obj = getattr(mod_obj, cls)
#         EVAL_OBJS.append(cls_obj(**kw))

# _CTX   = get_context("spawn")
# _POOL  = ProcessPoolExecutor(
#     max_workers=len(EVAL_CONFIG),
#     mp_context=_CTX,
#     initializer=_init_eval_worker  # 让子进程常驻 evaluator
# )

# def _run_eval(idx: int, gen_file: str, gold_file: str):
#     """在缓存的 Evaluator 上运行评测并返回 f1"""
#     ev = EVAL_OBJS[idx]
#     ev(generation_code_file=gen_file, golden_code_file=gold_file)
#     return ev.metrics.get("f1")

def _unique_prefix(plot_index: int) -> str:
    uid = f"{plot_index}_{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
    return os.path.join(os.environ['PROJECT_STORE_PATH'], uid)

def extract_python_blocks(text: str) -> list[str]:
    """
    从传入的 Markdown 字符串中提取所有以 ```python 开头、``` 结尾的代码块。
    返回一个去除首尾空白的代码片段列表。
    """
    pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    # code_blocks = '\n\n'.join(pattern.findall(text))
    code_blocks = pattern.findall(text)

    return code_blocks

def compute_score(model_output: str, ground_truth: dict) -> float:
    """返回 6 个 evaluator 的 F1 均值（缺省视为 0）"""
    if "```python" not in model_output:
        return 0.0

    gt_code  = '\n\n'.join(extract_python_blocks(ground_truth["answer"]))

    out_code = '\n\n'.join(extract_python_blocks(model_output))
    

    plot_index = int(ground_truth['index'])
    def _inject_save_to(code: str, save_path: str) -> str:
        # 清理 plt.savefig / show / close
        code = re.sub(r"plt\.savefig\(.*?\)\s*", "", code, flags=re.S)
        code = re.sub(r"plt\.show\(.*?\)\s*", "", code, flags=re.S)
        code = re.sub(r"fig\.savefig\(.*?\)\s*", "", code, flags=re.S)
        code = re.sub(r"plt\.close\(.*?\)\s*", "", code, flags=re.S)
        # 清理 plt.clf()
        code = re.sub(r"plt\.clf\(\)\s*", "", code, flags=re.S)
        code = code.replace("\'save_path\'", f"\'{save_path}\'").replace("\"save_path\"", f"\"{save_path}\"")
        match = _MPF_PLOT_PATTERN.search(code)
        if match:
            # 获取 savefig=... 前后的部分
            before_savefig = match.group(1)
            after_savefig = match.group(2)
            # 移除 savefig 参数
            code = code.replace(match.group(0), f"mpf.plot({before_savefig}{after_savefig})")
        
        # 在代码末尾添加 plt.savefig，让它统一处理保存
        code = code.strip() + '\nplt.savefig("{}")'.format(save_path)
        return code


    prefix = _unique_prefix(plot_index)
    gold_file = f"{prefix}_gt_org.py" 
    gen_file = f"{prefix}_generated_by_model_in_training.py"
    gold_png_file = f"{prefix}_gt_org.png"
    gen_png_file = f"{prefix}_generated_by_model_in_training.png"
    
    try:
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
    except Exception:
        pass
    gt_code_for_png = _inject_save_to(gt_code, gold_png_file)
    out_code_for_png = _inject_save_to(out_code, gen_png_file)
    with open(gold_file, 'w') as f:
        f.write(gt_code_for_png)
    with open(gen_file, 'w') as f:
        f.write(out_code_for_png)

    try:
        # add a timeout
        # subprocess.run(f"python {gen_file}", shell=True, timeout=60)
        execute_python_with_timeout(gen_file)
        if not os.path.exists(gen_png_file):
            os.remove(gen_file)
            os.remove(gold_file)
            return 0.0
    except subprocess.TimeoutExpired:
        print(f"Timeout: {gen_file}")
        os.remove(gen_file)
        os.remove(gold_file)
        return 0.0
    except Exception as e:
        print(f"Error: {gen_file} {e}")
        os.remove(gen_file)
        os.remove(gold_file)
        return 0.0


    scores = []
    text_evaluator = TextAttributeEvaluator(use_position=False, use_attributes=False, use_axs=True)
    chart_type_evaluator = ChartTypeEvaluator()
    color_evaluator = ColorEvaluator()
    layout_evaluator = LayoutEvaluator()
    data_evaluator = DataEvaluator()
    style_and_axes_evaluator = StyleAndAxesEvaluator()

    text_evaluator(gen_file, gold_file)
    chart_type_evaluator(gen_file, gold_file)
    color_evaluator(gen_file, gold_file)
    layout_evaluator(gen_file, gold_file)
    
    data_evaluator(gen_file, gold_file)
    # print(f'data_evaluator.metrics: {data_evaluator.metrics}')
    
    style_and_axes_evaluator(gen_file, gold_file)


    # print(f'style_and_axes_evaluator.metrics: {style_and_axes_evaluator.metrics}')
    # try:
    #     data_evaluator(gen_file, gold_file)
    # except Exception as e:
    #     data_evaluator.metrics["f1"] = 0.0

    # try:
    #     style_and_axes_evaluator(gen_file, gold_file)
    # except Exception as e:
    #     style_and_axes_evaluator.metrics["f1"] = 0.0

    try:
        scores.append(text_evaluator.metrics.get("f1"))
    except Exception as e:
        scores.append(0.0)

    try:
        scores.append(chart_type_evaluator.metrics.get("f1"))
    except Exception as e:
        scores.append(0.0)

    try:
        scores.append(color_evaluator.metrics.get("f1"))
    except Exception as e:
        scores.append(0.0)

    try:
        scores.append(layout_evaluator.metrics.get("f1"))
    except Exception as e:
        scores.append(0.0)

    try:
        scores.append(data_evaluator.metrics.get("f1"))
    except Exception as e:
        scores.append(0.0)

    try:
        scores.append(style_and_axes_evaluator.metrics.get("f1"))
    except Exception as e:
        scores.append(0.0)


    # # 提交到进程池并行执行（子进程内复用 Evaluator）
    # futures = [
    #     _POOL.submit(_run_eval, idx, gen_file, gold_file)
    #     for idx in range(len(EVAL_CONFIG))
    # ]
    # scores = []                  # 只保存非 None 的分数
    # for idx, fut in enumerate(futures):
    #     mod, cls, kw = EVAL_CONFIG[idx]
    #     score = fut.result()         # 只调用一次
    #     if score is None:
    #         print(f"{mod}-{cls}-{kw}: None")
    #         continue
    #     # if 'color' in mod:
    #     #     score = score * 1.5
    #     # elif 'layout' in mod:
    #     #     score = score * 0.5
    #     # elif 'text' in mod:
    #     #     score = score * 1.4
    #     # elif 'chart_type' in mod:
    #     #     score = score * 0.6
    #     scores.append(score)
    #     print(f"{mod}-{cls}-{kw}: {score:.4f}")   # 逐项打印

    # # 清理（GT 缓存保留）
    # for p in [gen_file, gold_file, gen_png_file, gold_png_file]:
    #     try: os.unlink(p)
    #     except FileNotFoundError: pass

    print(f"scores: {scores} // {float(np.mean(scores))}")
    os.remove(gen_file)
    os.remove(gold_file)
    os.remove(gen_png_file)
    os.remove(gold_png_file)

    return float(np.mean(scores)) if scores else 0.0