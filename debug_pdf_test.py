#!/usr/bin/env python3
"""
调试脚本：详细诊断PDF格式处理问题
"""

import os
import tempfile
import sys
import traceback

# 设置环境变量
os.environ["PROJECT_PACK_PATH"] = "/fs-computility/mllm1/fangxinyu/plot2code/verl"
os.environ["PROJECT_STORE_PATH"] = "/tmp"

def create_simple_pdf_code(filename):
    """创建一个简单的PDF生成代码"""
    code = f"""
import matplotlib.pyplot as plt
import numpy as np

# 创建简单图表
fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, 'PDF测试文字', ha='center', va='center', fontsize=16)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('PDF测试图')

# 保存为PDF
plt.savefig('{filename}.pdf', bbox_inches='tight')
plt.close()
"""
    return code

def test_pdf_generation():
    """测试PDF生成和文字提取的每个步骤"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"临时目录: {temp_dir}")
        
        # 创建测试代码
        pdf_code = create_simple_pdf_code(os.path.join(temp_dir, 'test'))
        pdf_file = os.path.join(temp_dir, 'test.py')
        with open(pdf_file, 'w', encoding='utf-8') as f:
            f.write(pdf_code)
        
        print("步骤1: 创建基础代码文件 ✅")
        print(f"代码文件: {pdf_file}")
        
        # 手动模拟TextEvaluator._log_texts的过程
        try:
            # 读取代码
            with open(pdf_file, 'r') as f:
                lines = f.readlines()
            code = ''.join(lines)
            print("步骤2: 读取代码内容 ✅")
            
            # 生成prefix
            prefix = f"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
sys.path.append('{os.environ['PROJECT_PACK_PATH']}')

# Import different renderers for various output formats
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

drawed_texts = []

def log_function(func):
    def wrapper(*args, **kwargs):
        global drawed_texts

        object = args[0]
        x = args[2]
        y = args[3]
        x_rel = ( x / object.width / 72 ) * 100
        y_rel = ( y / object.height / 72 ) * 100
        s = args[4]

        drawed_texts.append( (float(x), float(y), float(x_rel), float(y_rel), s) )
        return func(*args, **kwargs)

    return wrapper

# Hook multiple renderers to support different output formats
RendererPdf.draw_text = log_function(RendererPdf.draw_text)  # PDF format
RendererAgg.draw_text = log_function(RendererAgg.draw_text)  # PNG, JPG, etc.
if RendererSVG is not None:
    RendererSVG.draw_text = log_function(RendererSVG.draw_text)  # SVG format
if RendererPS is not None:
    RendererPS.draw_text = log_function(RendererPS.draw_text)  # PS/EPS format
"""
            
            output_file = pdf_file.replace(".py", "_log_texts.txt")
            suffix = f"""
# print("drawed_texts", drawed_texts)
with open('{output_file}', 'w') as f:
    f.write(str(drawed_texts))
"""
            
            # 组合完整代码
            full_code = prefix + code + suffix
            print("步骤3: 生成完整的hook代码 ✅")
            
            # 写入执行文件
            code_log_texts_file = pdf_file.replace(".py", "_log_texts.py")
            with open(code_log_texts_file, 'w') as f:
                f.write(full_code)
            print(f"步骤4: 创建执行文件 {code_log_texts_file} ✅")
            
            # 执行代码
            print("步骤5: 执行代码...")
            import subprocess
            result = subprocess.run([sys.executable, code_log_texts_file], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("步骤5: 代码执行成功 ✅")
                if result.stdout:
                    print(f"标准输出: {result.stdout}")
            else:
                print(f"步骤5: 代码执行失败 ❌")
                print(f"返回码: {result.returncode}")
                print(f"错误输出: {result.stderr}")
                print(f"标准输出: {result.stdout}")
                return
            
            # 检查输出文件
            if os.path.exists(output_file):
                print(f"步骤6: 输出文件创建成功 ✅ - {output_file}")
                with open(output_file, 'r') as f:
                    content = f.read()
                    print(f"文件内容: {content}")
                    try:
                        texts = eval(content)
                        print(f"解析后的文字: {texts}")
                        if texts:
                            print("✅ PDF文字提取成功!")
                        else:
                            print("⚠️  没有提取到文字，但文件格式正确")
                    except Exception as e:
                        print(f"❌ 解析文件内容失败: {e}")
            else:
                print(f"步骤6: 输出文件未创建 ❌ - {output_file}")
                
        except Exception as e:
            print(f"❌ 测试过程出错: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    print("开始PDF调试测试...")
    test_pdf_generation()
    print("调试测试完成！") 