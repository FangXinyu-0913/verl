import os
import re
import traceback
import subprocess
import re
from tqdm import tqdm
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ['PROJECT_PACK_PATH']='/fs-computility/mllm1/fangxinyu/plot2code/verl/verl/utils/reward_score/chart2code'
sys.path.insert(0,os.environ['PROJECT_PACK_PATH'])
os.environ['PROJECT_STORE_PATH']='/fs-computility/mllm1/shared/hub/datasets--xxxllz--Chart2Code-160k/processed_code/double_check_store'
from evaluator.text_evaluator import TextEvaluator
from evaluator.chart_type_evaluator import ChartTypeEvaluator
from evaluator.legend_evaluator import LegendEvaluator
from evaluator.grid_evaluator import GridEvaluator
from evaluator.color_evaluator import ColorEvaluator
from evaluator.layout_evaluator import LayoutEvaluator
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
import tempfile,importlib



savefig_pattern = re.compile(
    r"""
    plt\.savefig              # 关键字
    \(\s*                     # (
    (?P<quote>['"])           # 引号 '
    (?P<path>.*?)             # 原路径
    (?P=quote)                # 闭合引号
    (?P<rest>\s*,[^)]*)?      # 剩余参数（可选）
    \)                        # )
    """,
    re.VERBOSE | re.DOTALL,
)

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

def _run_eval(mod_path: str, cls_name: str, kwargs: dict,
              gen_file: str, gold_file: str):
    mod  = importlib.import_module(mod_path)
    cls  = getattr(mod, cls_name)
    ev   = cls(**kwargs)               # 实例化发生在子进程
    ev(generation_code_file=gen_file,
       golden_code_file=gold_file)
    return ev.metrics.get("f1")


def extract_python_blocks(text: str) -> list[str]:
    """
    从传入的 Markdown 字符串中提取所有以 ```python 开头、``` 结尾的代码块。
    返回一个去除首尾空白的代码片段列表。
    """
    pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    code_blocks = '\n\n'.join(pattern.findall(text))

    return code_blocks

def process_code_snippets(item: dict, output_dir: str = "/fs-computility/mllm1/shared/hub/datasets--xxxllz--Chart2Code-160k/processed_code") -> tuple[dict, bool]:
    """
    处理代码片段，执行以下操作：
    1. 将原始代码存储成一个 .py 文件，并检查其可运行性。
    2. 如果原始代码可运行，则修改代码：
        a. 将 plt.savefig 后面的存储路径提取出来，改成 id_val + '_generated.pdf'。
        b. 删除 plt.close()。
    3. 重新执行修改后的代码，如果可运行则保留修改后的代码。

    Args:
        item (dict): 包含 'conversations' 键的字典，其中包含原始的 GPT 输出。
        id_val (str): 用于文件命名和图片生成路径的唯一ID。
        output_dir (str): 保存处理后的代码文件的目录。

    Returns:
        tuple[dict, bool]: 一个元组，第一个元素是包含处理结果的字典（如果成功则包含 modified_code），
                            第二个元素是一个布尔值，表示操作是否全部成功 (True) 或失败 (False)。
    """
    try:
        org_gpt_output = item['conversations'][-1]['value']
        id_val = item['id']
    except (KeyError, IndexError) as e:
        print(f"[ID: {id_val}] 错误：{e}，无法从 item 中提取 conversations 或 value。")
        return {}, False
    
    original_code = extract_python_blocks(org_gpt_output)
    if not original_code:
        print(f"[ID: {id_val}] 未能从 GPT 输出中提取到 Python 代码块。")
        return {}, False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_filepath = os.path.join(output_dir, f"{id_val}_original.py")
    modified_filepath = os.path.join(output_dir, f"{id_val}_modified.py")
    
    results = {"id": id_val, "image": item['image'], "conversations": [item['conversations'][0]]}

    # 步骤 1: 将原始代码存储成一个py文件，并检查其可运行性
    try:
        with open(original_filepath, "w", encoding="utf-8") as f:
            f.write(original_code)
        
        print(f"[ID: {id_val}] 尝试运行原始代码...")
        run_result = subprocess.run(
            ["python", original_filepath],
            capture_output=True,
            text=True,
            timeout=60
        )

        if run_result.returncode != 0:
            print(f"[ID: {id_val}] 原始代码运行失败。错误：\n{run_result.stderr}")
            if os.path.exists(original_filepath):
                os.remove(original_filepath)
            return results, False
        
        print(f"[ID: {id_val}] 原始代码运行成功。")

    except Exception as e:
        print(f"[ID: {id_val}] 保存或运行原始代码时发生异常：{e}")
        if os.path.exists(original_filepath):
            os.remove(original_filepath)
        return results, False

    # 步骤 2 & 3: 修改代码 (savefig路径, 删除plt.close)
    modified_code = original_code

    # 步骤 2a: 留下有 plt.savefig 的，把 plt.savefig 后面的存储路径提取出来，改成 id_val + '_generated.pdf'
    if "plt.savefig" in modified_code:
        print(f"[ID: {id_val}] 找到 plt.savefig，正在修改路径...")
        def _replace(match: re.Match) -> str:
            quote = match.group('quote')
            rest  = match.group('rest') or ''
            return f"plt.savefig({quote}{id_val}_generated.pdf{quote}{rest})"

        # 只替换一次；n_subs 表示成功替换的次数
        modified_code, n_subs = savefig_pattern.subn(_replace, modified_code, count=1)
        
        if n_subs:
            print(f"[ID: {id_val}] plt.savefig 路径已修改为 '{id_val}_generated.pdf'")
        else:
            print(f"[ID: {id_val}] 未能修改 plt.savefig 路径。")
            return results, False
    else:
        print(f"[ID: {id_val}] 未找到 plt.savefig。此操作失败，因为要求保留有 plt.savefig 的。")
        # 如果没有 plt.savefig，并且我们要求只保留有它的，那么这里算作失败
        return results, False # 未找到 plt.savefig 视为失败

    # 步骤 2b: 将 plt.close() 进行删除 (无论是否存在都不算失败)
    if "plt.close()" in modified_code:
        print(f"[ID: {id_val}] 找到 plt.close()，正在删除...")
        modified_code = modified_code.replace("plt.close()", "")
        print(f"[ID: {id_val}] plt.close() 已删除。")
    else:
        print(f"[ID: {id_val}] 未找到 plt.close()。")

    # 保存修改后的代码
    try:
        with open(modified_filepath, "w", encoding="utf-8") as f:
            f.write(modified_code)
    except Exception as e:
        print(f"[ID: {id_val}] 保存修改后的代码时发生异常：{e}")
        return results, False
    
    # 步骤 3: 重新执行修改后的这个代码，保留能运行的
    try:
        print(f"[ID: {id_val}] 尝试运行修改后的代码...")
        run_result_modified = subprocess.run(
            ["python", modified_filepath],
            capture_output=True,
            text=True,
            timeout=60
        )

        if run_result_modified.returncode != 0:
            print(f"[ID: {id_val}] 修改后的代码运行失败。错误：\n{run_result_modified.stderr}")
            if os.path.exists(modified_filepath):
                os.remove(modified_filepath)
            return results, False
        
        print(f"[ID: {id_val}] 修改后的代码运行成功。")

    except Exception as e:
        print(f"[ID: {id_val}] 运行修改后的代码时发生异常：{e}")
        if os.path.exists(modified_filepath):
            os.remove(modified_filepath)
        return results, False
            
    # 如果所有步骤都成功，返回修改后的代码和 True
    results["conversations"].append({
        "from": "gpt",
        "value": "```python " + modified_code + " ```"
    })
    return results, True


import json

# LOAD & DUMP
def dump(data, f, **kwargs):

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False)

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')


    handlers = dict(json=dump_json, xlsx=dump_xlsx)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)


def load(f, fmt=None):

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    handlers = dict(json=load_json, jsonl=load_jsonl)
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split('.')[-1]
    return handlers[suffix](f)

def double_check(item, output_dir= "/fs-computility/mllm1/shared/hub/datasets--xxxllz--Chart2Code-160k/processed_code"):
    org_gpt_output = item['conversations'][-1]['value']
    id_val = item['id']
    original_code = extract_python_blocks(org_gpt_output)

    if not original_code:
        print(f"[ID: {id_val}] 未能从 GPT 输出中提取到 Python 代码块。")
        return False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_filepath = os.path.join(output_dir, f"{id_val}_original_doubleck.py")
    modified_filepath = os.path.join(output_dir, f"{id_val}_modified_doubleck.py")
    with open(original_filepath, "w", encoding="utf-8") as f:
        f.write(original_code)
    with open(modified_filepath, "w", encoding="utf-8") as f:
        f.write(original_code)


    futures = [
        _POOL.submit(_run_eval, mod, cls, kw, modified_filepath, original_filepath)
        for (mod, cls, kw) in EVAL_CONFIG
    ]
    scores = []                  # 只保存非 None 的分数
    for (mod, cls, kw), fut in zip(EVAL_CONFIG, futures):
        score = fut.result()         # 只调用一次
        if score is None:
            continue
        scores.append(score)
        print(f"{mod}-{cls}-{kw}: {score:.4f}")   # 逐项打印

    # 删除临时文件
    try: os.unlink(modified_filepath); os.unlink(original_filepath)
    except FileNotFoundError: pass

    if not scores:
        print(f"[ID: {id_val}] 所有评估指标均为 None，双重检查失败。")
        return False
    elif float(np.mean(scores)) == 1.0 and len(scores) == 4:
        print(f"[ID: {id_val}] 双重检查通过，所有评估指标均为 1。")
        return True
    else:
        print(f"[ID: {id_val}] 双重检查失败，评估指标均值为 {np.mean(scores)}，但不全为 1 或存在值缺失。")
        return False

    # TEXT_EVAL(generation_code_file=modified_filepath, golden_code_file=original_filepath)
    # LAYOUT_EVAL(generation_code_file=modified_filepath, golden_code_file=original_filepath)
    # CHART_EVAL(generation_code_file=modified_filepath, golden_code_file=original_filepath)
    # COLOR_EVAL(generation_code_file=modified_filepath, golden_code_file=original_filepath)


    # if os.path.exists(f'{id_val}_generated.pdf'):
    #     os.remove(f'{id_val}_generated.pdf')
    # f1_text = TEXT_EVAL.metrics.get('f1')
    # f1_layout = LAYOUT_EVAL.metrics.get('f1')
    # f1_chart_type = CHART_EVAL.metrics.get('f1')
    # f1_color = COLOR_EVAL.metrics.get('f1')

    # f1_list = [f1_text, f1_layout, f1_chart_type, f1_color]
    # try:
    #     os.remove(original_filepath)
    #     os.remove(modified_filepath)
    # except Exception as e:
    #     print(f"[ID: {id_val}] 删除临时文件时发生异常：{e}")

    # if all(f1 is not None and f1 == 1 for f1 in f1_list):
    #     print(f"[ID: {id_val}] 双重检查通过，所有评估指标均为 1。")
    #     return True
    # else:
    #     print(f"[ID: {id_val}] 双重检查失败")
    #     return False



if __name__ == "__main__":

    meta = load('/fs-computility/mllm1/shared/hub/datasets--xxxllz--Chart2Code-160k/chart2code_160k.json')
    # meta = meta[:int(len(meta) * 0.05)]  # 只处理前5%的数据

    def process_item(item):
        code, flag = process_code_snippets(item)
        if flag:
            double_check_flag = double_check(code)
            if flag and double_check_flag:
                return code, item['id']
            else:
                return None, item['id']
        else:
            return None, item['id']

    batch_size = 2000
    output_dir = '/fs-computility/mllm1/shared/hub/datasets--xxxllz--Chart2Code-160k/'
    output_prefix = 'chart2code_filtered_code_5p_batch'
    batch_index = 1
    while os.path.isfile(os.path.join(output_dir, f"{output_prefix}{batch_index}.json")):
        batch_index += 1
    batch_index -= 1
    ok_path = os.path.join(output_dir, f"{output_prefix}{batch_index}.json")
    filtered_data = load(ok_path) if os.path.exists(ok_path) else []
    last_id = filtered_data[-1]['id'] if os.path.exists(ok_path) else -1
    meta = [item for item in meta if item['id'] > last_id] if last_id != -1 else meta
    print(f"开始处理 {len(meta)} 条数据，已处理 {last_id} 条数据， 存储 {len(filtered_data)} 条数据。")


    with ThreadPoolExecutor(max_workers=18) as executor:
    # with ProcessPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_item, item): item for item in meta}
        for idx, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            code, item_id = future.result()
            try:
                os.remove(f"{item_id}_generated.pdf")
            except Exception as e:
                print(f"[ID: {item_id}] 删除生成的 PDF 文件时发生异常：{e}")
            if code:
                print(f"[ID: {item_id}] 处理成功，添加到结果列表。")
                filtered_data.append(code)
            # 每 batch_size 条保存一次
            if (len(filtered_data) > 0) and (len(filtered_data) % batch_size == 0):
                batch_index = len(filtered_data) // batch_size
                batch_path = f"{output_dir}{output_prefix}{batch_index}.json"
                dump(filtered_data, batch_path)
                print(f"已保存 {len(filtered_data)} 条数据到 {batch_path}")

    # 最后再保存一次，防止最后一批不到batch_size没存下来
    if filtered_data:
        batch_index = (len(filtered_data) - 1) // batch_size + 1
        batch_path = f"{output_dir}{output_prefix}{batch_index}_final.json"
        dump(filtered_data, batch_path)
        print(f"处理完成，共 {len(filtered_data)} 条数据成功处理，已全部保存到 {batch_path}。")


