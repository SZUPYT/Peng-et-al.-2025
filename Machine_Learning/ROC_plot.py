import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path # For easier path manipulation, especially for dummy file cleanup


# --- 1. 定义您的JSON文件路径 ---
# !!! 重要: 请将这里的示例路径替换为您真实的JSON文件路径 !!!
json_file_paths = [
    "/Users/lightman/Code/Github/Code/MachineLearning/New_Epoch/ml_visualizations/TOM_Rate/roc.json",
    "/Users/lightman/Code/Github/Code/MachineLearning/New_Epoch/ml_visualizations/TOM_Minus/roc.json",
    "/Users/lightman/Code/Github/Code/MachineLearning/New_Epoch/ml_visualizations/Rest_Relative/roc.json",
        "/Users/lightman/Code/Github/Code/MachineLearning/New_Epoch/ml_visualizations/Rest_Absolute/roc.json"
    # 例如: "output/run1/best_model_roc_params.json",
    #       "output/run2/best_model_roc_params.json",
    #       ...
]

# --- 2. 设置Matplotlib图表 ---
plt.figure(figsize=(10, 8)) # 可以根据需要调整图像大小
colors = ['darkorange', 'green', 'blue','red'] # 为不同曲线定义颜色
line_styles = ['-', '--', '-.', ':'] # 为不同曲线定义线型

# --- 3. 循环读取JSON文件并绘图 ---
print("--- 开始绘制组合ROC曲线 ---")
for i, file_path_str in enumerate(json_file_paths):
    file_path = Path(file_path_str)
    try:
        with open(file_path, 'r') as f:
            roc_data = json.load(f)

        # 从JSON数据中提取所需参数
        model_name = roc_data.get('model_name', f'Model {i+1}') # 如果没有model_name则使用默认名
        auc_score = roc_data.get('auc')
        fpr = roc_data.get('fpr')
        tpr = roc_data.get('tpr')
        
        # 获取CI值 (可能是数字, None, 或 float('nan'))
        auc_ci_lower = roc_data.get('auc_ci_lower')
        auc_ci_upper = roc_data.get('auc_ci_upper')

        if fpr is None or tpr is None or auc_score is None:
            print(f"  警告: 文件 '{file_path}' 缺少 fpr, tpr, 或 auc 数据。跳过此文件。")
            continue

        # 构建图例标签
        label = f'{model_name} (AUC = {auc_score:.2f}'
        # 检查CI值是否有效 (非None且非NaN)
        ci_lower_valid = auc_ci_lower is not None and not (isinstance(auc_ci_lower, float) and np.isnan(auc_ci_lower))
        ci_upper_valid = auc_ci_upper is not None and not (isinstance(auc_ci_upper, float) and np.isnan(auc_ci_upper))

        if ci_lower_valid and ci_upper_valid:
            label += f', 95% CI: [{auc_ci_lower:.2f}-{auc_ci_upper:.2f}]'
        label += ')'
        
        # 绘制ROC曲线
        plt.plot(fpr, tpr, color=colors[i % len(colors)], 
                 linestyle=line_styles[i % len(line_styles)], 
                 lw=2, label=label)
        print(f"  已添加来自 '{file_path}' 的曲线: {model_name}")
        
    except FileNotFoundError:
        print(f"  错误: 文件 '{file_path}' 未找到。请检查路径。")
    except json.JSONDecodeError:
        print(f"  错误: 无法解析文件 '{file_path}' 中的JSON数据。文件可能已损坏或格式不正确。")
    except Exception as e:
        print(f"  处理文件 '{file_path}' 时发生未知错误: {e}")

# --- 4. 添加图表元素 ---
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # 对角参考线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05]) # y轴略微延伸，确保曲线顶部可见
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity/Recall)')
plt.title('Comparison of Model ROC Curves')
plt.legend(loc="lower right", fontsize='small') # 图例放在右下角
plt.grid(alpha=0.4) # 添加浅色网格，可选

# --- 5. 显示或保存图表 ---
output_plot_path = "combined_roc_curves.png"
try:
    plt.savefig(output_plot_path, dpi=300)
    print(f"\n--- 组合ROC曲线图已保存至: {output_plot_path} ---")
except Exception as e:
    print(f"\n--- 保存组合ROC曲线图失败: {e} ---")

# plt.show() # 如果您的环境支持GUI，取消注释此行以显示图像
