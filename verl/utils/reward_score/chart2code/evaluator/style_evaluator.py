from typing import List, Dict, Any
import numpy as np
import os
import sys
import ast
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.spines
from .utils import execute_python_with_timeout

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

class StyleAndAxesEvaluator:
    # __init__, __call__, _log_styles, _calculate_metrics 保持不变
    def __init__(self, rtol=1e-5, atol=1e-8):
        self.metrics = {"f1": 0.0}; self.rtol = rtol; self.atol = atol
    # @with_timeout(timeout=120)
    def __call__(self, generation_code_file, golden_code_file):
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_gen = executor.submit(self._log_styles, generation_code_file)
            future_gold = executor.submit(self._log_styles, golden_code_file)
            generation_styles, golden_styles = future_gen.result(), future_gold.result()
        self._calculate_metrics(generation_styles, golden_styles)
    def _log_styles(self, code_file):
        with open(code_file, 'r', encoding='utf-8') as f: code = f.read()
        prefix = self._get_prefix_for_style_capture()
        output_file = code_file.replace(".py", "_log_styles.txt")
        suffix = self._get_suffix_for_styles(output_file)
        full_code = prefix + "\n" + code + "\n" + suffix
        code_log_styles_file = code_file.replace(".py", "_log_styles.py")
        with open(code_log_styles_file, 'w', encoding='utf-8') as f: f.write(full_code)
        execute_python_with_timeout(code_log_styles_file)
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data_str = f.read()
            # Wrap in try-except block for safety
            try:
                styles = ast.literal_eval(data_str) if data_str else []
            except (SyntaxError, ValueError):
                styles = [] # Return empty if parsing fails
            os.remove(output_file)
        else: styles = []
        if os.path.exists(code_log_styles_file): os.remove(code_log_styles_file)
        # 去掉styles里面properties为空的
        styles = [style for style in styles if style['properties']]
        return styles


    def _compare_props(self, props_gold, props_gen):
        """
        递归比较两个属性字典。
        处理浮点数容差，并对 int/float 类型进行兼容性比较。
        """
        # 如果都为None，则匹配
        if props_gold is None and props_gen is None:
            return True
        # 如果只有一者为None，则不匹配
        if props_gold is None or props_gen is None:
            return False

        # 如果类型不同，但都是数字，则使用 allclose
        is_numeric = lambda x: isinstance(x, (int, float, np.integer, np.floating))
        if is_numeric(props_gold) and is_numeric(props_gen):
            return np.allclose([props_gold], [props_gen], rtol=self.rtol, atol=self.atol)
        
        # 如果类型不同且不是数字，则不匹配
        if type(props_gold) != type(props_gen):
            return False

        # 字典类型，递归比较所有键值对
        if isinstance(props_gold, dict):
            # 确保键集合一致
            if sorted(props_gold.keys()) != sorted(props_gen.keys()):
                return False
            # 递归比较每个键对应的值
            return all(self._compare_props(props_gold[k], props_gen[k]) for k in props_gold)
        
        # 列表或元组类型，递归比较所有元素
        if isinstance(props_gold, (list, tuple)):
            # 确保长度一致
            if len(props_gold) != len(props_gen):
                return False
            # 递归比较每个元素
            return all(self._compare_props(g, p) for g, p in zip(props_gold, props_gen))
        
        # 字符串、布尔值等直接比较
        if isinstance(props_gold, (str, bool)):
            return props_gold == props_gen
        
        # 兜底：对于其他未知类型，直接比较
        return props_gold == props_gen

    def _calculate_metrics(self, generation_styles: List[Dict], golden_styles: List[Dict]):
        # print(f'StyleAndAxesEvaluator generation_styles: {generation_styles}')
        # print(f'StyleAndAxesEvaluator golden_styles: {golden_styles}')
        if not golden_styles:
            self.metrics["f1"] = 1.0 if not generation_styles else 0.0
            return
        if not generation_styles:
            self.metrics["f1"] = 0.0
            return

        # 创建 generation_styles 的可修改副本，用于匹配和移除
        gen_styles_copy = generation_styles.copy()
        
        true_positives = 0
        
        # 遍历 golden_styles，寻找匹配项
        for gold_item in golden_styles:
            best_match_idx = -1
            for i, gen_item in enumerate(gen_styles_copy):
                # 匹配条件
                is_match = False
                if (gold_item['type'] == gen_item['type'] and
                    gold_item.get('ax_id') == gen_item.get('ax_id')):
                    # 检查 data_hash (如果有的话)
                    if 'data_hash' in gold_item and gold_item.get('data_hash') != gen_item.get('data_hash'):
                        continue
                    # 比较 properties
                    if self._compare_props(gold_item['properties'], gen_item['properties']):
                        is_match = True
                
                if is_match:
                    best_match_idx = i
                    break
            
            # 如果找到匹配项，增加 TP 计数并移除副本中的该项，以避免重复匹配
            if best_match_idx != -1:
                true_positives += 1
                gen_styles_copy.pop(best_match_idx)

        # 计算查准率和查全率
        # TP = true_positives
        # FP = len(generation_styles) - true_positives
        # FN = len(golden_styles) - true_positives
        
        precision = true_positives / len(generation_styles) if len(generation_styles) > 0 else 0
        recall = true_positives / len(golden_styles) if len(golden_styles) > 0 else 0

        # 计算 F1 分数
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        self.metrics["f1"] = f1_score
    # def _compare_props(self, props_gold, props_gen):
    #     # ... (此方法保持不变) ...
    #     if props_gold is None and props_gen is None: return True
    #     if type(props_gold) != type(props_gen): return False
    #     if isinstance(props_gold, dict):
    #         if sorted(props_gold.keys()) != sorted(props_gen.keys()): return False
    #         return all(self._compare_props(props_gold[k], props_gen[k]) for k in props_gold)
    #     if isinstance(props_gold, (list, tuple)):
    #         if len(props_gold) != len(props_gen): return False
    #         return all(self._compare_props(g, p) for g, p in zip(props_gold, props_gen))
    #     if isinstance(props_gold, (int, str, bool)): return props_gold == props_gen
    #     if isinstance(props_gold, float): return np.allclose([props_gold], [props_gen], rtol=self.rtol, atol=self.atol)
    #     return props_gold == props_gen
    # def _calculate_metrics(self, generation_styles: List[Dict], golden_styles: List[Dict]):
    #     # ... (此方法保持不变) ...
    #     if not golden_styles:
    #         self.metrics["f1"] = 1.0 if not generation_styles else 0.0; return
    #     if not generation_styles:
    #         self.metrics["f1"] = 0.0; return
    #     score = 0; gen_styles_copy = generation_styles.copy()
    #     for gold_item in golden_styles:
    #         match_found = False
    #         for i, gen_item in enumerate(gen_styles_copy):
    #             if gold_item['type'] == gen_item['type'] and gold_item.get('ax_id') == gen_item.get('ax_id'):
    #                 if 'data_hash' in gold_item and gold_item.get('data_hash') != gen_item.get('data_hash'): continue
    #                 if self._compare_props(gold_item['properties'], gen_item['properties']):
    #                     match_found = True; gen_styles_copy.pop(i); break
    #         if match_found: score += 1
    #     self.metrics["f1"] = score / len(golden_styles) if len(golden_styles) > 0 else 1.0

    def _get_prefix_for_style_capture(self):
        return f"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import os
import matplotlib
# 设置非交互式后端，避免显示相关的资源竞争
os.environ['MPLBACKEND'] = 'Agg' 
matplotlib.use('Agg', force=True)
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互模式
import matplotlib.axes
import matplotlib.spines
import matplotlib.figure
import inspect, hashlib

FINAL_LOGGED_STYLES = []
LOGGED_ITEMS_HASHES = set()
# Human-readable ax_id mapping
AX_ID_MAP = {{}}
AX_COUNTER = 1

def _sanitize_for_eval(obj):
    if isinstance(obj, dict): return {{key: _sanitize_for_eval(value) for key, value in obj.items() if key != 'self'}}
    if isinstance(obj, (list, tuple)): return [_sanitize_for_eval(elem) for elem in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    # colormap -> name
    try:
        import matplotlib.colors as _mc
        if isinstance(obj, _mc.Colormap):
            return getattr(obj, "name", str(obj))
    except Exception:
        pass
    if isinstance(obj, (int, float, str, bool, type(None))): return obj
    return str(obj)

def to_list_safe(data):
    if data is None: return None
    if isinstance(data, np.ndarray): return data.tolist()
    if hasattr(data, 'values'): return data.values.tolist()
    try: return list(data)
    except TypeError: return str(data)

def _get_ax_id(args, kwargs):
    global AX_ID_MAP, AX_COUNTER
    ax_instance = None
    if args and isinstance(args[0], matplotlib.axes.Axes): ax_instance = args[0]
    elif 'ax' in kwargs and isinstance(kwargs['ax'], matplotlib.axes.Axes): ax_instance = kwargs['ax']
    if not ax_instance:
        try: ax_instance = plt.gca()
        except Exception: return "figure_level"
    raw_id = id(ax_instance)
    if raw_id not in AX_ID_MAP:
        AX_ID_MAP[raw_id] = f'subfigure_{{AX_COUNTER}}'
        AX_COUNTER += 1
    return AX_ID_MAP[raw_id]

def _log_item(item):
    sanitized_item = _sanitize_for_eval(item)
    item_str = str(sanitized_item)
    item_hash = hashlib.md5(item_str.encode()).hexdigest()
    if item_hash not in LOGGED_ITEMS_HASHES:
        FINAL_LOGGED_STYLES.append(sanitized_item)
        LOGGED_ITEMS_HASHES.add(item_hash)

# ---- wrappers ----
def set_lim_wrapper(original_func, lim_type):
    def wrapper(self, *args, **kwargs):
        props = {{}}
        ax_id = _get_ax_id((self,), {{}})
        if args:
            if isinstance(args[0], (tuple, list)): props['lim'] = args[0]
            elif len(args) >= 2: props['lim'] = (args[0], args[1])
        else: props = {{k: v for k, v in kwargs.items()}}
        _log_item({{'type': lim_type, 'ax_id': ax_id, 'properties': props}})
        return original_func(self, *args, **kwargs)
    return wrapper

def grid_wrapper(original_func):
    def wrapper(self, visible=None, **kwargs):
        props = {{'visible': visible, **kwargs}}
        if visible is None and not kwargs: props['visible'] = True
        if props.get('visible') is True:
            ax_id = _get_ax_id((self,), {{}})
            _log_item({{'type': 'grid', 'ax_id': ax_id, 'properties': props}})
        return original_func(self, visible=visible, **kwargs)
    return wrapper

def set_scale_wrapper(original_func, scale_type):
    def wrapper(self, value, **kwargs):
        ax_id = _get_ax_id((self,), {{}})
        _log_item({{'type': scale_type, 'ax_id': ax_id, 'properties': {{'scale': value, **kwargs}}}})
        return original_func(self, value, **kwargs)
    return wrapper

def set_aspect_wrapper(original_func):
    \"\"\"记录 set_aspect 的显式调用；仅当与默认不同才会在后处理阶段输出。\"\"\"
    def wrapper(self, aspect, *args, **kwargs):
        ax_id = _get_ax_id((self,), {{}})
        _log_item({{'type': 'aspect_set', 'ax_id': ax_id, 'properties': {{'aspect': aspect}}}})
        return original_func(self, aspect, *args, **kwargs)
    return wrapper

def spine_set_visible_wrapper(original_func):
    def wrapper(self, value):
        if value is False:
            ax_id = None
            for ax in plt.gcf().get_axes():
                if self in ax.spines.values(): ax_id = _get_ax_id((ax,), {{}}); break
            _log_item({{'type': 'spine_visibility', 'ax_id': ax_id, 'properties': {{'spine': self.spine_type, 'visible': value}}}})
        return original_func(self, value)
    return wrapper

def axline_wrapper(original_func, line_type):
    def wrapper(self, *args, **kwargs):
        bound = inspect.signature(original_func).bind(self, *args, **kwargs)
        bound.apply_defaults()
        ax_id = _get_ax_id((self,), {{}})
        _log_item({{'type': line_type, 'ax_id': ax_id, 'properties': dict(bound.arguments)}})
        return original_func(self, *args, **kwargs)
    return wrapper

def axspan_wrapper(original_func, span_type):
    def wrapper(self, *args, **kwargs):
        bound = inspect.signature(original_func).bind(self, *args, **kwargs)
        bound.apply_defaults()
        ax_id = _get_ax_id((self,), {{}})
        _log_item({{'type': span_type, 'ax_id': ax_id, 'properties': dict(bound.arguments)}})
        return original_func(self, *args, **kwargs)
    return wrapper

def general_plot_style_wrapper(original_func, plot_type, data_keys):
    def wrapper(*args, **kwargs):
        ax_id = _get_ax_id(args, kwargs)
        style_props = {{k: v for k, v in kwargs.items() if k not in data_keys and k != 'ax'}}
        # 让 imshow 的 cmap 记录为名字字符串
        if plot_type == 'imshow' and 'cmap' in style_props:
            try:
                style_props['cmap'] = getattr(style_props['cmap'], 'name', str(style_props['cmap']))
            except Exception:
                pass
        _log_item({{'type': f'{{plot_type}}_style', 'ax_id': ax_id, 'properties': style_props}})
        return original_func(*args, **kwargs)
    return wrapper

def legend_wrapper(original_func):
    def wrapper(*args, **kwargs):
        legend_obj = original_func(*args, **kwargs)
        ax_id = _get_ax_id(args, kwargs)
        props = {{}}
        if legend_obj:
            props['loc'] = legend_obj._get_loc()
            props['ncols'] = legend_obj._ncols
            if args and isinstance(args[0], (list, tuple)) and all(isinstance(i, str) for i in args[0]):
                props['labels_from_args'] = args[0]
        for k in ['loc', 'ncol', 'fontsize', 'bbox_to_anchor']:
            if k in kwargs: props[k] = kwargs[k]
        _log_item({{'type': 'legend_style', 'ax_id': ax_id, 'properties': props}})
        return legend_obj
    return wrapper

def figure_wrapper(original_func):
    def wrapper(*args, **kwargs):
        if 'figsize' in kwargs:
            _log_item({{'type': 'figsize', 'properties': {{'figsize': kwargs.get('figsize')}}}})
        return original_func(*args, **kwargs)
    return wrapper

# NEW: colorbar wrapper —— 记录 ticks，并关联到其 mappable 的 Axes
def colorbar_wrapper(original_func):
    def wrapper(*args, **kwargs):
        cb = original_func(*args, **kwargs)
        try:
            mapp = getattr(cb, 'mappable', None)
            ax = getattr(mapp, 'axes', None)
            ax_id = _get_ax_id((ax,), {{}}) if ax is not None else 'figure_level'
            ticks = [float(t) for t in cb.get_ticks()]
            if ticks:
                _log_item({{'type':'colorbar','ax_id': ax_id,'properties':{{'ticks':ticks}}}})
        except Exception:
            pass
        return cb
    return wrapper

# --- Patch methods ---
matplotlib.axes.Axes.set_xlim = set_lim_wrapper(matplotlib.axes.Axes.set_xlim, 'xlim')
matplotlib.axes.Axes.set_ylim = set_lim_wrapper(matplotlib.axes.Axes.set_ylim, 'ylim')
matplotlib.axes.Axes.set_xscale = set_scale_wrapper(matplotlib.axes.Axes.set_xscale, 'xscale')
matplotlib.axes.Axes.set_yscale = set_scale_wrapper(matplotlib.axes.Axes.set_yscale, 'yscale')
matplotlib.axes.Axes.set_aspect = set_aspect_wrapper(matplotlib.axes.Axes.set_aspect)
matplotlib.axes.Axes.grid = grid_wrapper(matplotlib.axes.Axes.grid)
matplotlib.spines.Spine.set_visible = spine_set_visible_wrapper(matplotlib.spines.Spine.set_visible)

matplotlib.axes.Axes.axvline = axline_wrapper(matplotlib.axes.Axes.axvline, 'axvline')
matplotlib.axes.Axes.axhline = axline_wrapper(matplotlib.axes.Axes.axhline, 'axhline')
matplotlib.axes.Axes.axvspan = axspan_wrapper(matplotlib.axes.Axes.axvspan, 'axvspan')
matplotlib.axes.Axes.axhspan = axspan_wrapper(matplotlib.axes.Axes.axhspan, 'axhspan')

matplotlib.axes.Axes.legend = legend_wrapper(matplotlib.axes.Axes.legend)
plt.legend = legend_wrapper(plt.legend)

plot_data_keys = ['x', 'y', 's', 'c', 'self', 'data', 'X']
matplotlib.axes.Axes.plot = general_plot_style_wrapper(matplotlib.axes.Axes.plot, 'plot', plot_data_keys)
matplotlib.axes.Axes.scatter = general_plot_style_wrapper(matplotlib.axes.Axes.scatter, 'scatter', plot_data_keys)
matplotlib.axes.Axes.bar = general_plot_style_wrapper(matplotlib.axes.Axes.bar, 'bar', ['x','height','self'])
matplotlib.axes.Axes.barh = general_plot_style_wrapper(matplotlib.axes.Axes.barh, 'barh', ['y','width','self'])
matplotlib.axes.Axes.hist = general_plot_style_wrapper(matplotlib.axes.Axes.hist, 'hist', ['x','self'])
matplotlib.axes.Axes.pie = general_plot_style_wrapper(matplotlib.axes.Axes.pie, 'pie', ['x','self'])
matplotlib.axes.Axes.errorbar = general_plot_style_wrapper(matplotlib.axes.Axes.errorbar, 'errorbar', plot_data_keys)
matplotlib.axes.Axes.imshow = general_plot_style_wrapper(matplotlib.axes.Axes.imshow, 'imshow', ['X','self'])

# Figure/pyplot level
plt.figure = figure_wrapper(plt.figure)
plt.subplots = figure_wrapper(plt.subplots)
plt.colorbar = colorbar_wrapper(plt.colorbar)
matplotlib.figure.Figure.colorbar = colorbar_wrapper(matplotlib.figure.Figure.colorbar)
"""

    def _get_suffix_for_styles(self, output_file):
        return f"""
# --- Post-processing logged styles for compression & extras ---
import numpy as _np, matplotlib as _mpl, matplotlib.pyplot as _plt

processed_styles = []
latest_settings = {{}}  # last one wins for unique props
unique_types = ['xlim', 'ylim', 'xscale', 'yscale']

for item in FINAL_LOGGED_STYLES:
    t = item.get('type'); ax_id = item.get('ax_id')
    if t in unique_types:
        latest_settings[(ax_id, t)] = item
    else:
        processed_styles.append(item)
processed_styles.extend(latest_settings.values())

# ===== Extra 1: imshow 的 origin/aspect/vmin/vmax（仅在“实际使用/非默认”时记录） =====
fig = _plt.gcf()
rc_origin_default = _mpl.rcParams.get('image.origin', 'upper')
rc_aspect_default = _mpl.rcParams.get('image.aspect', 'equal')  # imshow 的默认行为参照 rc
for ax in fig.get_axes():
    ax_id = None
    try: 
        ax_id = [k for k,v in AX_ID_MAP.items() if v==v]  # noqa: keep mapping alive
        ax_id = AX_ID_MAP.get(id(ax), None)
    except Exception:
        ax_id = None
    if ax_id is None:
        continue

    # 找轴上的最后一个 AxesImage
    images = getattr(ax, 'images', [])
    if images:
        im = images[-1]
        extras = {{}}

        # origin：仅当与 rcParams 默认不同才记录
        try:
            origin = getattr(im, 'origin', None)
            if origin is not None and origin != rc_origin_default:
                extras['origin'] = origin
        except Exception:
            pass

        # aspect：若 rc 默认是 'equal' 而当前为 'auto'（常见热力图设置），才记录
        try:
            aspect = ax.get_aspect()
            if rc_aspect_default == 'equal' and aspect == 'auto':
                extras['aspect'] = 'auto'
        except Exception:
            pass

        # vmin/vmax：只有用户显式设定时 norm.vmin/vmax 才不是 None
        try:
            norm = getattr(im, 'norm', None)
            vmin = getattr(norm, 'vmin', None); vmax = getattr(norm, 'vmax', None)
            if vmin is not None and vmax is not None:
                processed_styles.append({{'type':'heatmap_color','ax_id': ax_id, 'properties': {{'vmin': float(vmin), 'vmax': float(vmax)}}}})
        except Exception:
            pass

        if extras:
            processed_styles.append({{'type':'imshow_extras','ax_id': ax_id, 'properties': extras}})

# ===== Extra 2: colorbar 已在 wrapper 中记录 ticks；此处无需重复 =====

# ===== Extra 3: 自动侦测格内注释（annot/fmt） =====
# 规则：若该 Axes 有 AxesImage，且有 >=60% 的文本位于 (j,i) 网格中心，并且文本全是整数
for ax in fig.get_axes():
    ax_id = AX_ID_MAP.get(id(ax), None)
    if ax_id is None: 
        continue
    images = getattr(ax, 'images', [])
    if not images:
        continue
    im = images[-1]
    arr = _np.array(im.get_array()) if hasattr(im, 'get_array') else None
    if arr is None or arr.ndim != 2:
        continue
    H, W = arr.shape
    texts = [t for t in ax.texts if t.get_visible()]
    if not texts:
        continue

    def _is_int_str(s):
        s = str(s).strip()
        if s.startswith(('+','-')): s = s[1:]
        # 纯数字（不含小数点）
        return s.isdigit()

    hit = 0
    for t in texts:
        x, y = t.get_position()
        # 近整数、且位于网格范围内
        if abs(x - round(x)) < 0.25 and abs(y - round(y)) < 0.25:
            j, i = int(round(x)), int(round(y))
            if 0 <= i < H and 0 <= j < W and t.get_ha()=='center' and t.get_va()=='center' and _is_int_str(t.get_text()):
                hit += 1
    if hit >= 0.6 * H * W:
        processed_styles.append({{'type':'heatmap_annot','ax_id': ax_id, 'properties': {{'annot': True, 'fmt': 'd'}}}})

# 只保留 properties 非空的项
processed_styles = [s for s in processed_styles if s.get('properties')]

with open('{output_file}', 'w', encoding='utf-8') as f:
    f.write(str(processed_styles))
"""
