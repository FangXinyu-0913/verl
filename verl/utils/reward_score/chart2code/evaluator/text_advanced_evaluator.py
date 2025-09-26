from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()
import os
import sys
# 假设您的 utils 模块路径设置正确
# sys.path.append(os.environ["PROJECT_PACK_PATH"])
from .utils import execute_python_with_timeout 

# --- 为了独立运行，先定义一个假的 execute_python_with_timeout ---
# def execute_python_with_timeout(filepath):
#     try:
#         os.system(f'python {filepath}')
#     except Exception as e:
#         print(f"Failed to execute {filepath}: {e}")
# -----------------------------------------------------------------

import ast
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.legend
from matplotlib.text import Text
from matplotlib.backends.backend_pdf import RendererPdf
from matplotlib.backends.backend_agg import RendererAgg
try:
    from matplotlib.backends.backend_svg import RendererSVG
except ImportError:
    RendererSVG = None
try:
    from matplotlib.backends.backend_ps import RendererPS
except ImportError:
    RendererPS = None

class TextAttributeEvaluator:

    def __init__(self, use_position=False, use_attributes=False, use_axs=True) -> None:
        self.metrics = {"precision": 0, "recall": 0, "f1": 0}
        self.use_position = use_position
        self.use_attributes = use_attributes
        self.use_axs = use_axs

    def __call__(self, generation_code_file, golden_code_file):
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_gen = executor.submit(self._log_texts, generation_code_file)
            future_gold = executor.submit(self._log_texts, golden_code_file)
            
            generation_texts = future_gen.result()
            golden_texts = future_gold.result()

        self._calculate_metrics(generation_texts, golden_texts)
        
        # Clean up generated PDF if it exists in a specific path
        if "PROJECT_STORE_PATH" in os.environ:
            redundant_file = os.path.join(os.environ["PROJECT_STORE_PATH"], os.path.basename(golden_code_file).replace(".py", ".pdf"))
            if os.path.exists(redundant_file):
                os.remove(redundant_file)


    def _log_texts(self, code_file):
        with open(code_file, 'r', encoding='utf-8') as f:
            code = f.read()

        prefix = self._get_prefix_with_attributes()
        output_file = code_file.replace(".py", "_log_texts.txt")
        suffix = self._get_suffix(output_file)
        
        # 清理原脚本中的保存/显示/关闭，避免重复 I/O 放大，新加入，不知道有没有用
        import re as _re
        try:
            code = _re.sub(r"plt\\.savefig\(.*?\)\s*", "", code, flags=_re.S)
            code = _re.sub(r"fig\\.savefig\(.*?\)\s*", "", code, flags=_re.S)
            code = _re.sub(r"plt\\.show\(.*?\)\s*", "", code, flags=_re.S)
            code = _re.sub(r"plt\\.close\(.*?\)\s*", "", code, flags=_re.S)
            code = _re.sub(r"plt\\.clf\(.*?\)\s*", "", code, flags=_re.S)
        except Exception:
            pass

        code = prefix + "\n" + code + "\n" + suffix

        if not self.use_axs:
            savefig_idx = code.find("plt.savefig")
            if savefig_idx != -1:
                ax_ticks_deletion_code = self._get_ax_ticks_deletion_code()
                code = code[:savefig_idx] + ax_ticks_deletion_code + code[savefig_idx:]

        code_log_texts_file = code_file.replace(".py", "_log_texts.py")
        # 若已有输出，直接读取，避免重复执行
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    texts_str = f.read()
                    try:
                        texts = ast.literal_eval(texts_str)
                    except (ValueError, SyntaxError):
                        texts = []
                os.remove(output_file)
                return texts
            except Exception:
                pass

        with open(code_log_texts_file, 'w', encoding='utf-8') as f:
            f.write(code)
            
        execute_python_with_timeout(code_log_texts_file)

        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                texts_str = f.read()
                try:
                    texts = ast.literal_eval(texts_str)
                except (ValueError, SyntaxError):
                    texts = []
            os.remove(output_file)
        else:
            texts = []
        
        if os.path.exists(code_log_texts_file):
            os.remove(code_log_texts_file)

        return texts

    def _calculate_metrics(self, generation_texts: List[Dict], golden_texts: List[Dict]):
        try:
            if not generation_texts or not golden_texts:
                self.metrics = {"precision": 0, "recall": 0, "f1": 0}
                return

            len_generation = len(generation_texts)
            len_golden = len(golden_texts)

            n_correct = 0
            temp_generation_texts = generation_texts[:]

            for g_text in golden_texts:
                for i, gen_text in enumerate(temp_generation_texts):
                    if g_text['text'] == gen_text['text'] and g_text['type'] == gen_text['type']:
                        position_match = True
                        if self.use_position:
                            # CHANGED: Update index for relative position (2 for x, 3 for y)
                            pos_g_rel_x, pos_g_rel_y = g_text['position'][2], g_text['position'][3]
                            pos_gen_rel_x, pos_gen_rel_y = gen_text['position'][2], gen_text['position'][3]
                            if not (abs(pos_g_rel_x - pos_gen_rel_x) <= 10 and abs(pos_g_rel_y - pos_gen_rel_y) <= 10):
                                position_match = False

                        attribute_match = True
                        if self.use_attributes:
                            size_g = g_text['attributes'].get('fontsize')
                            size_gen = gen_text['attributes'].get('fontsize')
                            if size_g is not None and size_gen is not None and abs(size_g - size_gen) > 2:
                                attribute_match = False
                        
                        if position_match and attribute_match:
                            n_correct += 1
                            temp_generation_texts.pop(i)
                            break
                
            self.metrics["precision"] = n_correct / len_generation if len_generation > 0 else 0
            self.metrics["recall"] = n_correct / len_golden if len_golden > 0 else 0
            
            if self.metrics["precision"] + self.metrics["recall"] == 0:
                self.metrics["f1"] = 0
            else:
                self.metrics["f1"] = 2 * self.metrics["precision"] * self.metrics["recall"] / (self.metrics["precision"] + self.metrics["recall"])

        except Exception:
            self.metrics["f1"] = None
            
    def _get_prefix_with_attributes(self):
        return f"""
import warnings
warnings.filterwarnings("ignore")

import sys, os
import matplotlib
# 设置非交互式后端，避免显示相关的资源竞争
os.environ['MPLBACKEND'] = 'Agg' 
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互模式
import matplotlib.axes
import matplotlib.legend
from matplotlib.text import Text
from matplotlib.backends.backend_pdf import RendererPdf
from matplotlib.backends.backend_agg import RendererAgg
try:
    from matplotlib.backends.backend_svg import RendererSVG
except ImportError:
    RendererSVG = None
try:
    from matplotlib.backends.backend_ps import RendererPS
except ImportError:
    RendererPS = None

# --- Global Data Stores ---
TEXT_OBJECT_REGISTRY = {{}}
FINAL_LOGGED_TEXTS = []
LOGGED_OBJECT_IDS = set()  # 去重

def extract_text_properties(text_obj: Text) -> dict:
    if not isinstance(text_obj, Text): return {{}}
    color = text_obj.get_color()
    if hasattr(color, 'tolist'): color = color.tolist()
    return {{
        'fontsize': text_obj.get_fontsize(),
        'fontfamily': text_obj.get_fontfamily(),
        'fontweight': text_obj.get_fontweight(),
        'color': color,
        'alpha': text_obj.get_alpha(),
        'rotation': text_obj.get_rotation(),
    }}

# 低层渲染钩子：记录 obj_id、位置、属性
def low_level_draw_text_hook(original_func):
    def wrapper(renderer, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        global FINAL_LOGGED_TEXTS, TEXT_OBJECT_REGISTRY, LOGGED_OBJECT_IDS
        if mtext and id(mtext) in LOGGED_OBJECT_IDS:
            return original_func(renderer, gc, x, y, s, prop, angle, ismath=ismath, mtext=mtext)

        if mtext:
            LOGGED_OBJECT_IDS.add(id(mtext))
            text_type = TEXT_OBJECT_REGISTRY.get(id(mtext), 'axis labels')

            width, height = renderer.get_canvas_width_height()
            x_rel = (x / width) * 100 if width > 0 else 0.0
            y_rel = (y / height) * 100 if height > 0 else 0.0

            entry = {{
                'obj_id': int(id(mtext)),
                'text': s,
                'type': text_type,
                'position': (float(x), float(y), float(x_rel), float(y_rel)),
                'attributes': extract_text_properties(mtext)
            }}
            FINAL_LOGGED_TEXTS.append(entry)

        return original_func(renderer, gc, x, y, s, prop, angle, ismath=ismath, mtext=mtext)
    return wrapper

# 高层登记：标题/轴名/图内标注/图例
def high_level_text_wrapper(original_func, text_type):
    def wrapper(*args, **kwargs):
        obj = original_func(*args, **kwargs)
        if isinstance(obj, Text):
            TEXT_OBJECT_REGISTRY[id(obj)] = text_type
        return obj
    return wrapper

def legend_wrapper(original_func):
    def wrapper(*args, **kwargs):
        lg = original_func(*args, **kwargs)
        if lg:
            if lg.get_title() and lg.get_title().get_text():
                TEXT_OBJECT_REGISTRY[id(lg.get_title())] = 'legend_title'
            for t in lg.get_texts():
                TEXT_OBJECT_REGISTRY[id(t)] = 'legend_label'
        return lg
    return wrapper

# 应用补丁
RendererPdf.draw_text = low_level_draw_text_hook(RendererPdf.draw_text)
RendererAgg.draw_text = low_level_draw_text_hook(RendererAgg.draw_text)
if RendererSVG: RendererSVG.draw_text = low_level_draw_text_hook(RendererSVG.draw_text)
if RendererPS:  RendererPS.draw_text  = low_level_draw_text_hook(RendererPS.draw_text)

plt.title   = high_level_text_wrapper(plt.title,   'title')
plt.xlabel  = high_level_text_wrapper(plt.xlabel,  'xlabel')
plt.ylabel  = high_level_text_wrapper(plt.ylabel,  'ylabel')
plt.suptitle= high_level_text_wrapper(plt.suptitle,'suptitle')
plt.text    = high_level_text_wrapper(plt.text,    'in_figure_data_label')

matplotlib.axes.Axes.set_title = high_level_text_wrapper(matplotlib.axes.Axes.set_title, 'title')
matplotlib.axes.Axes.set_xlabel= high_level_text_wrapper(matplotlib.axes.Axes.set_xlabel,'xlabel')
matplotlib.axes.Axes.set_ylabel= high_level_text_wrapper(matplotlib.axes.Axes.set_ylabel,'ylabel')
matplotlib.axes.Axes.text      = high_level_text_wrapper(matplotlib.axes.Axes.text,     'in_figure_data_label')

plt.legend = legend_wrapper(plt.legend)
matplotlib.axes.Axes.legend = legend_wrapper(matplotlib.axes.Axes.legend)
"""


    def _get_suffix(self, output_file):
        return f"""
# === 在保存前，把 tick 文本从 'axis labels' 细分为 'row_label' / 'col_label' ===
import matplotlib.pyplot as _plt

try:
    fig = _plt.gcf()
    # 收集需要改型的 obj_id 集合
    col_ids = set()
    row_ids = set()

    for ax in fig.get_axes():
        # 仅在“可视化主轴”上分类（有图像的轴，避免 colorbar）
        has_images = bool(getattr(ax, 'images', []))
        if not has_images:
            continue

        for t in ax.get_xticklabels():
            if t.get_text():
                col_ids.add(int(id(t)))
        for t in ax.get_yticklabels():
            if t.get_text():
                row_ids.add(int(id(t)))

    # 就地更新 FINAL_LOGGED_TEXTS 的类型
    for item in FINAL_LOGGED_TEXTS:
        oid = item.get('obj_id')
        if not oid or item.get('type') != 'axis labels':
            continue
        if oid in col_ids:
            item['type'] = 'col_label'
        elif oid in row_ids:
            item['type'] = 'row_label'

except Exception:
    pass

with open('{output_file}', 'w', encoding='utf-8') as f:
    f.write(str(FINAL_LOGGED_TEXTS))
"""


    def _get_ax_ticks_deletion_code(self):
        return """
try:
    all_axes = plt.gcf().get_axes()
    for ax in all_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
except Exception:
    pass
"""