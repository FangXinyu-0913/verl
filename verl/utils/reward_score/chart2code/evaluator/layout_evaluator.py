from typing import Any, Dict, List
import os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import ast               # 比 eval 安全
import textwrap

import matplotlib.pyplot as plt
from dotenv import load_dotenv
import time
load_dotenv()

sys.path.append(os.environ["PROJECT_PACK_PATH"])

class LayoutEvaluator:
    def __init__(self) -> None:
        self.metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # ===========================  公共入口  ===========================
    def __call__(self, generation_code_file: str, golden_code_file: str):
        with ThreadPoolExecutor(max_workers=2) as tp:
            gen_future  = tp.submit(self._log_layouts, generation_code_file)
            gold_future = tp.submit(self._log_layouts, golden_code_file)
            gen_layouts, gold_layouts = gen_future.result(), gold_future.result()
        self._calculate_metrics(gen_layouts, gold_layouts)

        # 清理冗余 PDF
        pdf = Path(os.environ["PROJECT_STORE_PATH"]) / Path(golden_code_file).with_suffix(".pdf").name
        if pdf.exists():
            pdf.unlink()

    # ==========================  主功能块  ============================
    def _log_layouts(self, code_file: str) -> List[Dict[str, Any]]:
        """执行 <原脚本 + 收集布局的后缀>，解析得到布局信息"""
        try:
            with open(code_file, "r", encoding="utf-8") as f:
                original_code = f.read()

            output_txt = code_file.replace(".py", "_log_layouts.txt")
            # 清理原脚本中的保存/显示/关闭，避免重复 I/O 放大
            import re as _re
            try:
                original_code = _re.sub(r"plt\\.savefig\(.*?\)\s*", "", original_code, flags=_re.S)
                original_code = _re.sub(r"fig\\.savefig\(.*?\)\s*", "", original_code, flags=_re.S)
                original_code = _re.sub(r"plt\\.show\(.*?\)\s*", "", original_code, flags=_re.S)
                original_code = _re.sub(r"plt\\.close\(.*?\)\s*", "", original_code, flags=_re.S)
                original_code = _re.sub(r"plt\\.clf\(.*?\)\s*", "", original_code, flags=_re.S)
            except Exception:
                pass

            probe_code = self._get_prefix() + original_code + self._get_suffix(output_txt)

            probe_path = code_file.replace(".py", "_log_layouts.py")
            with open(probe_path, "w", encoding="utf-8") as f:
                f.write(probe_code)

            # os.system(f"python3 {probe_path}")
            from .utils import execute_python_with_timeout
            # 若已有输出，直接读取，避免重复执行
            if Path(output_txt).exists():
                try:
                    with open(output_txt, "r", encoding="utf-8") as f:
                        try:
                            layouts = ast.literal_eval(f.read())
                        except Exception:
                            layouts = []
                    Path(output_txt).unlink()
                    return layouts
                except Exception:
                    pass

            execute_python_with_timeout(probe_path)  # 使用封装的执行函数，支持超时

            # 读取并解析输出
            if Path(output_txt).exists():
                with open(output_txt, "r", encoding="utf-8") as f:
                    try:
                        layouts = ast.literal_eval(f.read())  # 比 eval 更安全
                    except Exception:
                        layouts = []
                Path(output_txt).unlink()
            else:
                layouts = []

            Path(probe_path).unlink()
        except Exception as e:
            print(f"Error occurred while running the _log_layouts: {e}")
            print(f"Error output: {e.stderr}")
            layouts = []
        return layouts

    def _calculate_metrics(self, gen: List[Dict[str, Any]], gold: List[Dict[str, Any]]):
        if not gen or not gold:
            self.metrics = dict.fromkeys(self.metrics, 0.0)
            return

        freeze = lambda d: tuple(sorted(d.items()))
        gen_set, gold_set = set(map(freeze, gen)), set(map(freeze, gold))

        correct = len(gen_set & gold_set)
        prec = correct / len(gen_set)
        rec  = correct / len(gold_set)
        f1   = 0.0 if not (prec or rec) else 2 * prec * rec / (prec + rec)
        self.metrics.update(precision=prec, recall=rec, f1=f1)

    # ===========================  辅助代码  ===========================
    @staticmethod
    def _get_prefix() -> str:
        return textwrap.dedent(f"""
            import warnings, sys
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            sys.path.append('{os.environ["PROJECT_PACK_PATH"]}')
            import matplotlib
            import os
            # 设置非交互式后端，避免显示相关的资源竞争
            os.environ['MPLBACKEND'] = 'Agg' 
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
            plt.ioff()  # 关闭交互模式

            _orig_clf   = plt.clf
            _orig_close = plt.close

            def _no_clear(*a, **kw): pass
            plt.clf  = _no_clear               # 禁止清空
            plt.close = _no_clear              # 禁止关闭
        """)

    @staticmethod
    def _get_suffix(output_file: str) -> str:
        """任何 matplotlib 图都能识别；无 GridSpec 时默认 1×1"""
        return textwrap.dedent(f"""
            import matplotlib.pyplot as plt

            def get_layout(fig):
                info = {{}}
                for ax in fig.axes:
                    spec = getattr(ax, "get_subplotspec", lambda: None)()
                    if spec is None:
                        # 没用 GridSpec：认为整幅图是 1×1
                        info[ax] = dict(nrows=1, ncols=1,
                                        row_start=0, row_end=0,
                                        col_start=0, col_end=0)
                        continue

                    gs = spec.get_gridspec()
                    nrows, ncols = gs.get_geometry()
                    info[ax] = dict(
                        nrows=nrows, ncols=ncols,
                        row_start=spec.rowspan.start,
                        row_end  =spec.rowspan.stop - 1,
                        col_start=spec.colspan.start,
                        col_end  =spec.colspan.stop - 1,
                    )
                return list(info.values())

            layouts = get_layout(plt.gcf())
            with open('{output_file}', 'w', encoding='utf-8') as f:
                f.write(repr(layouts))
        """)
