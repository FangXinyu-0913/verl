from typing import List, Dict, Any
import numpy as np
import os
import sys
import ast
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.axes

# 假设您的 utils 模块路径设置正确
# sys.path.append(os.environ["PROJECT_PACK_PATH"])
from .utils import execute_python_with_timeout

import numpy as np
from typing import Dict, List, Any

import concurrent.futures
import functools

def with_timeout(timeout):
    """
    装饰器：给类的 __call__ 方法加上超时限制
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"{func.__qualname__} 超过 {timeout} 秒未完成，已超时！")
        return wrapper
    return decorator

# 核心依赖：确保已安装 scikit-learn 和 scipy
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import jensenshannon
except ImportError:
    print("请先安装依赖: pip install numpy scikit-learn scipy")
    # 如果没有安装，可以定义一个优雅降级的函数
    cosine_similarity = None
    jensenshannon = None

def calculate_similarity(data_gold: Dict, data_gen: Dict, plot_type: str) -> float:
    """
    计算两个图表数据元素的“软”相似度分数（0.0到1.0之间）。

    Args:
        data_gold (Dict): 黄金标准的数据字典。
        data_gen (Dict): 生成的数据字典。
        plot_type (str): 图表类型，如 'plot', 'hist', 'pie' 等。

    Returns:
        float: 0.0到1.0之间的相似度分数。
    """
    # 如果依赖没有正确导入，返回0
    if cosine_similarity is None or jensenshannon is None:
        return 0.0

    try:
        # --- 1. 适用于大多数基于X-Y坐标的图表 ---
        if plot_type in ['plot', 'scatter', 'bar', 'barh', 'errorbar', 'plot_knots']:
            # 根据图表类型确定关键的数据字段
            if plot_type == 'barh':
                gold_vals = np.array(data_gold.get('width', []))
                gen_vals = np.array(data_gen.get('width', []))
            elif plot_type == 'bar':
                gold_vals = np.array(data_gold.get('height', []))
                gen_vals = np.array(data_gen.get('height', []))
            else: # plot, scatter, errorbar, etc.
                gold_vals = np.array(data_gold.get('y', []))
                gen_vals = np.array(data_gen.get('y', []))

            # 预检查：长度必须一致且不为空
            if len(gold_vals) != len(gen_vals) or len(gold_vals) == 0:
                return 0.0

            # 计算余弦相似度（需要2D输入）
            score = cosine_similarity(gold_vals.reshape(1, -1), gen_vals.reshape(1, -1))[0, 0]
            
            # 将分数从[-1, 1]映射到[0, 1]
            return (score + 1) / 2

        # --- 2. 适用于直方图 (Histogram) ---
        elif plot_type == 'hist':
            gold_counts = np.array(data_gold.get('counts', []))
            gen_counts = np.array(data_gen.get('counts', []))

            # 预检查：长度必须一致且不为空
            if len(gold_counts) != len(gen_counts) or len(gold_counts) == 0:
                return 0.0
            
            # 归一化为概率分布
            sum_gold = np.sum(gold_counts)
            sum_gen = np.sum(gen_counts)
            
            if sum_gold == 0 or sum_gen == 0:
                return 1.0 if sum_gold == sum_gen else 0.0

            p = gold_counts / sum_gold
            q = gen_counts / sum_gen
            
            # 计算JS距离 (结果在0-1之间，0表示完全相同)
            distance = jensenshannon(p, q)
            # 将距离转换为相似度
            return 1.0 - distance

        # --- 3. 适用于饼图 (Pie Chart) ---
        elif plot_type == 'pie':
            gold_sizes = np.array(data_gold.get('sizes', []))
            gen_sizes = np.array(data_gen.get('sizes', []))
            gold_labels = data_gold.get('labels', [])
            gen_labels = data_gen.get('labels', [])

            # 预检查：sizes长度必须一致且不为空
            if len(gold_sizes) != len(gen_sizes) or len(gold_sizes) == 0:
                return 0.0

            # Part A: 计算数值(sizes)的相似度
            size_score = cosine_similarity(gold_sizes.reshape(1, -1), gen_sizes.reshape(1, -1))[0, 0]
            size_similarity = (size_score + 1) / 2
            
            # Part B: 计算标签(labels)的相似度 (Jaccard Similarity)
            if gold_labels is None or gen_labels is None:
                # 如果一方没有标签，则仅当两方都没有标签时才算匹配
                label_similarity = 1.0 if gold_labels is None and gen_labels is None else 0.0
            else:
                set_gold = set(gold_labels)
                set_gen = set(gen_labels)
                intersection = len(set_gold.intersection(set_gen))
                union = len(set_gold.union(set_gen))
                label_similarity = intersection / union if union != 0 else 1.0
            
            # 加权平均：数值权重70%，标签权重30%
            return 0.7 * size_similarity + 0.3 * label_similarity

        # --- 4. 适用于热力图 (Heatmap) ---
        elif plot_type == 'heatmap':
            gold_values = np.array(data_gold.get('values', []))
            gen_values = np.array(data_gen.get('values', []))

            # 预检查：形状必须完全一致
            if gold_values.shape != gen_values.shape or gold_values.size == 0:
                return 0.0
            
            # 将2D矩阵展平为1D向量，再计算余弦相似度
            score = cosine_similarity(
                gold_values.flatten().reshape(1, -1),
                gen_values.flatten().reshape(1, -1)
            )[0, 0]
            return (score + 1) / 2
            
        # --- 5. 适用于简单的单值比较 (axvline) ---
        elif plot_type == 'axvline':
            def single_value_similarity(val_gold, val_gen, epsilon=1e-9):
                # 使用相对误差来计算相似度
                denominator = abs(val_gold) + abs(val_gen) + epsilon
                similarity = 1.0 - (abs(val_gold - val_gen) / denominator)
                return max(0, similarity) #确保不为负

            sim_x = single_value_similarity(data_gold.get('x'), data_gen.get('x'))
            sim_ymin = single_value_similarity(data_gold.get('ymin'), data_gen.get('ymin'))
            sim_ymax = single_value_similarity(data_gold.get('ymax'), data_gen.get('ymax'))
            
            # 取三个参数相似度的平均值
            return (sim_x + sim_ymin + sim_ymax) / 3.0

        # --- 如果类型未覆盖，返回0 ---
        else:
            return 0.0

    except (KeyError, ValueError, TypeError):
        # 捕获任何因数据结构不匹配或类型错误导致的异常
        return 0.0

class DataEvaluator:

    def __init__(self, rtol=1e-5, atol=1e-8, max_points_threshold=60) -> None:
        self.metrics = {"f1": 0.0}
        self.rtol = rtol
        self.atol = atol
        self.max_points_threshold = max_points_threshold

    # @with_timeout(timeout=120)
    def __call__(self, generation_code_file, golden_code_file):
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_gen = executor.submit(self._log_data, generation_code_file)
            future_gold = executor.submit(self._log_data, golden_code_file)
            
            generation_data = future_gen.result()
            golden_data = future_gold.result()
        # print(f"generation_data: {generation_data}")
        # print(f"golden_data: {golden_data}")

        self._calculate_metrics(generation_data, golden_data)

    def _log_data(self, code_file):
        with open(code_file, 'r', encoding='utf-8') as f:
            code = f.read()

        prefix = self._get_prefix_for_data_capture()
        output_file = code_file.replace(".py", "_log_data.txt")
        suffix = self._get_suffix_for_data(output_file)
        
        # 清理原脚本中的保存/显示/关闭，避免重复 I/O 放大, 新加入，不知道有没有用
        import re as _re
        try:
            code = _re.sub(r"plt\\.savefig\(.*?\)\s*", "", code, flags=_re.S)
            code = _re.sub(r"fig\\.savefig\(.*?\)\s*", "", code, flags=_re.S)
            code = _re.sub(r"plt\\.show\(.*?\)\s*", "", code, flags=_re.S)
            code = _re.sub(r"plt\\.close\(.*?\)\s*", "", code, flags=_re.S)
            code = _re.sub(r"plt\\.clf\(.*?\)\s*", "", code, flags=_re.S)
        except Exception:
            pass

        full_code = prefix + "\n" + code + "\n" + suffix

        code_log_data_file = code_file.replace(".py", "_log_data.py")
        # 若已有输出，直接读取，避免重复执行
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data_str = f.read()
                    try:
                        data = ast.literal_eval(data_str)
                    except (ValueError, SyntaxError):
                        data = []
                os.remove(output_file)
                return data
            except Exception:
                pass

        with open(code_log_data_file, 'w', encoding='utf-8') as f:
            f.write(full_code)
            
        execute_python_with_timeout(code_log_data_file)

        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data_str = f.read()
                try: data = ast.literal_eval(data_str)
                except (ValueError, SyntaxError): data = []
            os.remove(output_file)
        else:
            data = []
        
        if os.path.exists(code_log_data_file): os.remove(code_log_data_file)

        return data

    def _calculate_metrics(self, generation_data: List[Dict], golden_data: List[Dict]):
        # 如果黄金数据为空，根据生成数据是否为空来决定 F1 分数
        # print(f'DataEvaluator generation_data: {generation_data}')
        # print(f'DataEvaluator golden_data: {golden_data}')
        if not golden_data:
            self.metrics["f1"] = 1.0 if not generation_data else 0.0
            return
        
        # 如果生成数据为空，F1 分数为 0
        if not generation_data:
            self.metrics["f1"] = 0.0
            return

        # 创建 generation_data 的可修改副本，用于匹配和移除
        gen_items_copy = generation_data.copy()
        
        true_positives = 0
        
        # 遍历 golden_data 中的每一个黄金标准项

        for gold_item in golden_data:
            best_match_idx = -1
            highest_similarity_score = -1.0
            similarity_threshold = 0.7  # 您可以调整这个阈值

            # 在 generation_data 的副本中寻找最佳匹配项
            for i, gen_item in enumerate(gen_items_copy):
                # 类型必须匹配
                if gold_item.get('type') != gen_item.get('type'):
                    continue
                
                # 调用我们新的相似度函数
                current_score = calculate_similarity(
                    gold_item.get('data', {}),
                    gen_item.get('data', {}),
                    gold_item.get('type')
                )
                try:
                    if current_score > highest_similarity_score:
                        highest_similarity_score = current_score
                        best_match_idx = i
                except Exception as e:
                    print(f"Error calculating similarity: {e}, {current_score}, {highest_similarity_score}")
                    continue
            
            # 如果找到的最佳匹配分数高于阈值，则视为一个真阳性
            if highest_similarity_score >= similarity_threshold:
                true_positives += 1
                # 【可选】对于软F1分数，可以这样：
                # true_positives += highest_similarity_score 
                gen_items_copy.pop(best_match_idx)

            

        # 计算查准率和查全率
        precision = true_positives / len(generation_data) if len(generation_data) > 0 else 0
        recall = true_positives / len(golden_data) if len(golden_data) > 0 else 0

        # 计算 F1 分数
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        self.metrics["f1"] = f1_score

    def _get_prefix_for_data_capture(self):
        return f"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import matplotlib
# 设置非交互式后端，避免显示相关的资源竞争
os.environ['MPLBACKEND'] = 'Agg' 
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互模式
import matplotlib.axes
import inspect

MAX_POINTS_THRESHOLD = {self.max_points_threshold}
FIT_DENSE_PLOTS_DEGREE = None

FINAL_LOGGED_DATA = []

def to_list_safe(data):
    if data is None: return None
    if isinstance(data, np.ndarray): return data.tolist()
    # Special handling for pandas DataFrames if they appear
    if hasattr(data, 'values'):
        return data.values.tolist()
    try: return list(data)
    except TypeError: return data

def _is_duplicate(new_item, existing_items):
    for item in existing_items:
        if new_item['type'] == item['type'] and new_item.get('ax_id') == item.get('ax_id'):
            try:
                data_new, data_old = new_item['data'], item['data']
                if sorted(data_new.keys()) != sorted(data_old.keys()): continue
                all_values_match = True
                for key in data_new:
                    val_new, val_old = data_new[key], data_old[key]
                    if isinstance(val_new, list):
                        if not np.array_equal(val_new, val_old): all_values_match = False; break
                    elif val_new != val_old: all_values_match = False; break
                if all_values_match: return True
            except: continue
    return False

def _get_ax_id(args):
    if args and isinstance(args[0], matplotlib.axes.Axes): return id(args[0])
    try: return id(plt.gca())
    except: return None

def _resolve_param_name(param_obj):
    if param_obj is None:
        return None
    # 优先使用 .name 属性 (例如 matplotlib Colormap 对象)
    if hasattr(param_obj, 'name') and isinstance(param_obj.name, str):
        return param_obj.name
    # 对于无法直接序列化的其他对象，转换为字符串
    if not isinstance(param_obj, (str, int, float, bool)):
        return str(param_obj)
    # 对于基本类型和 None，保持原样
    return param_obj

# ... (all previous wrappers: hist, bar, plot, scatter, errorbar, pie) ...
def hist_wrapper(original_func):
    def wrapper(*args, **kwargs):
        res = original_func(*args, **kwargs); n, bins, patches = res if res else (None, None, None)
        if n is None: return res
        new_item = {{'type': 'hist', 'ax_id': _get_ax_id(args), 'data': {{'counts': to_list_safe(n), 'bins': to_list_safe(bins)}}, 'params': {{'label': _resolve_param_name(kwargs.get('label'))}}}}
        if not _is_duplicate(new_item, FINAL_LOGGED_DATA): FINAL_LOGGED_DATA.append(new_item)
        return res
    return wrapper
def bar_wrapper(original_func):
    def wrapper(*args, **kwargs):
        try:
            bound_args = inspect.signature(original_func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            x = bound_args.arguments.get('x')
            height = bound_args.arguments.get('height')

            if x is not None and height is not None:
                new_item = {{'type': 'bar', 'ax_id': _get_ax_id(args), 'data': {{'x': to_list_safe(x), 'height': to_list_safe(height)}}, 'params': {{'label': _resolve_param_name(kwargs.get('label'))}}}}
                if not _is_duplicate(new_item, FINAL_LOGGED_DATA):
                    FINAL_LOGGED_DATA.append(new_item)
        except Exception:
            pass # Ignore if we cannot parse arguments
        return original_func(*args, **kwargs)
    return wrapper
# FIX 2: NEW Wrapper for barh (horizontal bars)
def barh_wrapper(original_func):
    def wrapper(*args, **kwargs):
        try:
            bound_args = inspect.signature(original_func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            y = bound_args.arguments.get('y')
            width = bound_args.arguments.get('width')

            if y is not None and width is not None:
                new_item = {{'type': 'barh', 'ax_id': _get_ax_id(args), 'data': {{'y': to_list_safe(y), 'width': to_list_safe(width)}}, 'params': {{'label': _resolve_param_name(kwargs.get('label'))}}}}
                if not _is_duplicate(new_item, FINAL_LOGGED_DATA):
                    FINAL_LOGGED_DATA.append(new_item)
        except Exception:
            pass
        return original_func(*args, **kwargs)
    return wrapper
# === New: dense-curve compression & classification ===
DENSE_CURVE_STRATEGY = "auto"   # "auto"|"gaussian_only"|"downsample_only"
GAUSS_RELERR_THRESH = 0.035     # 相对误差阈值，越小越严格
DOWNSAMPLE_KNOTS    = 40        # 下采样最多保留的点数

def _try_fit_gaussian(x, y):

    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    # 过滤非法点
    m = np.isfinite(x) & np.isfinite(y) & (y >= 0)
    x, y = x[m], y[m]
    if x.size < 10 or np.all(y == 0): 
        return False, {{}}
    # 初值（加权均值/方差）
    A0  = float(np.max(y))
    mu0 = float(np.sum(x*y) / (np.sum(y) + 1e-12))
    s0  = float(np.sqrt(np.sum(y*(x-mu0)**2) / (np.sum(y) + 1e-12)))
    if not np.isfinite(s0) or s0 <= 1e-6:
        s0 = (np.max(x)-np.min(x))/8.0

    def err(A, mu, s):
        yhat = A * np.exp(- (x-mu)**2 / (2.0*s*s + 1e-12))
        denom = np.maximum(np.max(y), 1e-8)
        return float(np.sqrt(np.mean((y-yhat)**2)) / denom)

    # 粗搜：在 mu0±s0、s∈[0.5s0,2.5s0] 上网格搜索
    mu_grid = np.linspace(mu0 - s0, mu0 + s0, 9)
    s_grid  = np.linspace(max(0.2*s0, 1e-3), 2.5*s0, 9)
    best = (1e9, A0, mu0, s0)
    for mu in mu_grid:
        for s in s_grid:
            # 闭式近似 A：使得 A*exp(...) 在峰值点与 y 的比值接近
            # 这里用最小二乘近似（A = <y,phi>/<phi,phi>）
            phi = np.exp(- (x-mu)**2 / (2.0*s*s + 1e-12))
            A = float(np.sum(y*phi) / (np.sum(phi*phi) + 1e-12))
            e = err(A, mu, s)
            if e < best[0]:
                best = (e, A, mu, s)

    rel_err, A1, mu1, s1 = best
    ok = np.isfinite([A1, mu1, s1]).all() and rel_err <= GAUSS_RELERR_THRESH
    return ok, {{"amp": float(A1), "mu": float(mu1), "sigma": float(abs(s1)), 
                "domain": [float(np.min(x)), float(np.max(x))], "rel_err": float(rel_err)}}

def _downsample_xy(x, y, k=DOWNSAMPLE_KNOTS):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size <= k: 
        return x.tolist(), y.tolist()
    # 累积弧长
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s  = np.concatenate([[0.0], np.cumsum(ds)])
    s_targets = np.linspace(0.0, s[-1], k)
    x_new = np.interp(s_targets, s, x)
    y_new = np.interp(s_targets, s, y)
    return x_new.tolist(), y_new.tolist()

# ---- Replace: plot_wrapper (dense curve auto-compress) ----
def plot_wrapper(original_func):
    def wrapper(*args, **kwargs):
        data_args = list(args)[1:] if (args and isinstance(args[0], matplotlib.axes.Axes)) else list(args)
        x, y = [], []
        if len(data_args) == 1:
            y = np.asarray(data_args[0]); x = np.arange(len(y))
        elif len(data_args) >= 2 and 'data' not in kwargs:
            x, y = data_args[0], data_args[1]

        logged = False
        if hasattr(x, '__len__') and hasattr(y, '__len__') and not isinstance(x, str) and not isinstance(y, str):
            if MAX_POINTS_THRESHOLD is not None and len(x) > MAX_POINTS_THRESHOLD:
                if DENSE_CURVE_STRATEGY in ("auto","gaussian_only"):
                    ok, params = _try_fit_gaussian(x, y)
                    if ok:
                        new_item = {{
                            'type': 'plot_gaussian', 'ax_id': _get_ax_id(args),
                            'data': {{'amp': params['amp'], 'mu': params['mu'], 'sigma': params['sigma'], 
                                     'domain': params['domain']}},
                            'params': {{'label': _resolve_param_name(kwargs.get('label'))}}
                        }}
                        if not _is_duplicate(new_item, FINAL_LOGGED_DATA):
                            FINAL_LOGGED_DATA.append(new_item)
                            logged = True
                if not logged and DENSE_CURVE_STRATEGY in ("auto","downsample_only"):
                    xs, ys = _downsample_xy(x, y, DOWNSAMPLE_KNOTS)
                    new_item = {{
                        'type': 'plot_knots', 'ax_id': _get_ax_id(args),
                        'data': {{'x': xs, 'y': ys}},
                        'params': {{'label': _resolve_param_name(kwargs.get('label'))}}
                    }}
                    if not _is_duplicate(new_item, FINAL_LOGGED_DATA):
                        FINAL_LOGGED_DATA.append(new_item)
                        logged = True
            if not logged:
                new_item = {{
                    'type': 'plot', 'ax_id': _get_ax_id(args),
                    'data': {{'x': to_list_safe(x), 'y': to_list_safe(y)}},
                    'params': {{'label': _resolve_param_name(kwargs.get('label'))}}
                }}
                if not _is_duplicate(new_item, FINAL_LOGGED_DATA):
                    FINAL_LOGGED_DATA.append(new_item)

        return original_func(*args, **kwargs)
    return wrapper

def scatter_wrapper(original_func):
    def wrapper(*args, **kwargs):
        data_args = list(args)[1:] if (args and isinstance(args[0], matplotlib.axes.Axes)) else list(args); x, y = data_args[0], data_args[1]
        new_item = {{'type': 'scatter', 'ax_id': _get_ax_id(args), 'data': {{'x': to_list_safe(x), 'y': to_list_safe(y)}}, 'params': {{'label': _resolve_param_name(kwargs.get('label'))}}}}
        if not _is_duplicate(new_item, FINAL_LOGGED_DATA): FINAL_LOGGED_DATA.append(new_item)
        return original_func(*args, **kwargs)
    return wrapper
def errorbar_wrapper(original_func):
    def wrapper(*args, **kwargs):
        data_args = list(args)[1:] if (args and isinstance(args[0], matplotlib.axes.Axes)) else list(args); x, y = data_args[0], data_args[1]
        new_item = {{'type': 'errorbar', 'ax_id': _get_ax_id(args), 'data': {{'x': to_list_safe(x), 'y': to_list_safe(y), 'yerr': to_list_safe(kwargs.get('yerr')), 'xerr': to_list_safe(kwargs.get('xerr'))}}, 'params': {{'label': _resolve_param_name(kwargs.get('label'))}}}}
        if not _is_duplicate(new_item, FINAL_LOGGED_DATA): FINAL_LOGGED_DATA.append(new_item)
        return original_func(*args, **kwargs)
    return wrapper
def pie_wrapper(original_func):
    def wrapper(*args, **kwargs):
        res = original_func(*args, **kwargs); data_args = list(args)[1:] if (args and isinstance(args[0], matplotlib.axes.Axes)) else list(args); sizes = data_args[0]
        new_item = {{'type': 'pie', 'ax_id': _get_ax_id(args), 'data': {{'sizes': to_list_safe(sizes), 'labels': to_list_safe(kwargs.get('labels'))}}, 'params': {{}}}}
        if not _is_duplicate(new_item, FINAL_LOGGED_DATA): FINAL_LOGGED_DATA.append(new_item)
        return res
    return wrapper

# NEW: Wrapper for heatmap (imshow)
def imshow_wrapper(original_func):
    def wrapper(*args, **kwargs):
        # The main data matrix 'X' is the first argument
        data_args = list(args)[1:] if (args and isinstance(args[0], matplotlib.axes.Axes)) else list(args)
        X = data_args[0]
        
        # Check if X looks like a 2D numerical array, not an image read from file (which would have 3 channels)
        # FIX 1: Handle both NumPy array and Pandas DataFrame for dtype checking
        is_numerical_2d = False
        if hasattr(X, 'ndim') and X.ndim == 2:
            if hasattr(X, 'dtype'): # Case for NumPy array
                is_numerical_2d = np.issubdtype(X.dtype, np.number)
            elif hasattr(X, 'dtypes'): # Case for Pandas DataFrame
                # Check if all columns are numeric
                is_numerical_2d = all(np.issubdtype(dt, np.number) for dt in X.dtypes)
        
        if is_numerical_2d:
            new_item = {{
                'type': 'heatmap',
                'ax_id': _get_ax_id(args),
                'data': {{
                    'values': to_list_safe(X)
                }},
                'params': {{
                    'cmap': _resolve_param_name(kwargs.get('cmap')),
                    'vmin': _resolve_param_name(kwargs.get('vmin')),
                    'vmax': _resolve_param_name(kwargs.get('vmax'))
                }}
            }}
            if not _is_duplicate(new_item, FINAL_LOGGED_DATA):
                FINAL_LOGGED_DATA.append(new_item)

        return original_func(*args, **kwargs)
    return wrapper

# ---- Robust axvline wrapper (handles both plt.axvline and Axes.axvline) ----
def axvline_wrapper(original_func):
    import inspect
    def wrapper(*args, **kwargs):
        try:
            # 统一用签名绑定来取参数（适配 method/function，拿到默认值）
            sig = inspect.signature(original_func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            # Axes 方法时，bound.arguments 会包含 'self'
            x = bound.arguments.get('x', None)
            ymin = float(bound.arguments.get('ymin', 0.0))
            ymax = float(bound.arguments.get('ymax', 1.0))
        except Exception:
            # 兜底：尽量从 args / kwargs 提取
            if kwargs.get('x', None) is not None:
                x = kwargs['x']
            else:
                # method: args[0]=Axes, args[1]=x; function: args[0]=x
                if len(args) >= 2 and isinstance(args[0], matplotlib.axes.Axes):
                    x = args[1]
                elif len(args) >= 1:
                    x = args[0]
                else:
                    x = 0.0
            ymin = float(kwargs.get('ymin', 0.0))
            ymax = float(kwargs.get('ymax', 1.0))

        # 有些场景 x 可能是 numpy 标量或可转换对象
        try:
            x_val = float(np.asarray(x).astype(float))
        except Exception:
            # 若仍不可转，放弃记录但不影响原函数执行
            return original_func(*args, **kwargs)

        new_item = {{
            'type': 'axvline',
            'ax_id': _get_ax_id(args),
            'data': {{'x': x_val, 'ymin': ymin, 'ymax': ymax}},
        }}
        if not _is_duplicate(new_item, FINAL_LOGGED_DATA):
            FINAL_LOGGED_DATA.append(new_item)

        return original_func(*args, **kwargs)
    return wrapper

# 应用补丁（注意两处都要挂）
plt.axvline = axvline_wrapper(plt.axvline)
matplotlib.axes.Axes.axvline = axvline_wrapper(matplotlib.axes.Axes.axvline)



# Applying all patches
# ... (all previous patches)
plt.hist = hist_wrapper(plt.hist); matplotlib.axes.Axes.hist = hist_wrapper(matplotlib.axes.Axes.hist)
plt.bar = bar_wrapper(plt.bar); matplotlib.axes.Axes.bar = bar_wrapper(matplotlib.axes.Axes.bar)
plt.plot = plot_wrapper(plt.plot); matplotlib.axes.Axes.plot = plot_wrapper(matplotlib.axes.Axes.plot)
plt.scatter = scatter_wrapper(plt.scatter); matplotlib.axes.Axes.scatter = scatter_wrapper(matplotlib.axes.Axes.scatter)
plt.errorbar = errorbar_wrapper(plt.errorbar); matplotlib.axes.Axes.errorbar = errorbar_wrapper(matplotlib.axes.Axes.errorbar)
plt.pie = pie_wrapper(plt.pie); matplotlib.axes.Axes.pie = pie_wrapper(matplotlib.axes.Axes.pie)
plt.barh = barh_wrapper(plt.barh)
matplotlib.axes.Axes.barh = barh_wrapper(matplotlib.axes.Axes.barh)
# ADDED: Applying patch for imshow
plt.imshow = imshow_wrapper(plt.imshow)
matplotlib.axes.Axes.imshow = imshow_wrapper(matplotlib.axes.Axes.imshow)
"""

    def _get_suffix_for_data(self, output_file):
        return f"""
with open('{output_file}', 'w', encoding='utf-8') as f:
    f.write(str(FINAL_LOGGED_DATA))
"""