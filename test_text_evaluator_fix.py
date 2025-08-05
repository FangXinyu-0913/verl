#!/usr/bin/env python3
"""
测试脚本：验证TextEvaluator修复后能否正确处理PNG和PDF格式
"""

import os
import tempfile
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from verl.utils.reward_score.chart2code.evaluator.text_evaluator import TextEvaluator

def create_test_code(filename, format='png'):
    """创建测试用的matplotlib代码"""
    code = f"""
import matplotlib.pyplot as plt
import numpy as np

# 创建一个简单的图表
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)

ax.plot(x, y, label='sin(x)')
ax.set_xlabel('X轴标签')
ax.set_ylabel('Y轴标签')  
ax.set_title('测试图表标题')
ax.legend()
ax.grid(True)

# 保存为指定格式
plt.savefig('{filename}.{format}', dpi=150, bbox_inches='tight')
plt.close()
"""
    return code

def test_text_evaluator():
    """测试TextEvaluator对不同格式的支持"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建PNG格式的测试代码
        png_code = create_test_code(os.path.join(temp_dir, 'test_png'), 'png')
        png_file = os.path.join(temp_dir, 'test_png.py')
        with open(png_file, 'w', encoding='utf-8') as f:
            f.write(png_code)
            
        # 创建PDF格式的测试代码  
        pdf_code = create_test_code(os.path.join(temp_dir, 'test_pdf'), 'pdf')
        pdf_file = os.path.join(temp_dir, 'test_pdf.py')
        with open(pdf_file, 'w', encoding='utf-8') as f:
            f.write(pdf_code)
            
        # 创建evaluator实例
        evaluator = TextEvaluator(use_position=False, use_axs=True)
        
        print("测试PNG格式文字提取...")
        try:
            # 测试PNG格式
            evaluator(png_file, pdf_file)  # 使用PNG vs PDF来测试
            print(f"PNG测试完成，metrics: {evaluator.metrics}")
            
            # 检查是否成功提取到文字
            if evaluator.metrics['f1'] is not None and evaluator.metrics['f1'] > 0:
                print("✅ PNG格式文字提取成功！")
            else:
                print("❌ PNG格式文字提取失败")
                
        except Exception as e:
            print(f"❌ PNG测试出错: {e}")
            
        print("\n测试PDF格式文字提取...")
        try:
            # 重新初始化evaluator
            evaluator = TextEvaluator(use_position=False, use_axs=True)
            # 测试PDF格式
            evaluator(pdf_file, pdf_file)  # 使用相同文件来测试基本功能
            print(f"PDF测试完成，metrics: {evaluator.metrics}")
            
            # 检查是否成功提取到文字
            if evaluator.metrics['f1'] is not None and evaluator.metrics['f1'] >= 0.9:  # 相同文件应该有很高的F1分数
                print("✅ PDF格式文字提取成功！")
            else:
                print("❌ PDF格式文字提取失败")
                
        except Exception as e:
            print(f"❌ PDF测试出错: {e}")

if __name__ == "__main__":
    print("开始测试TextEvaluator修复...")
    test_text_evaluator()
    print("测试完成！") 