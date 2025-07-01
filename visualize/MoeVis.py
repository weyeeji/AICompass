import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches

model_name = 'mixtral-8x7B'

# 创建pic文件夹（如果不存在）
if not os.path.exists('pic'):
    os.makedirs('pic')

# 读取CSV文件 - 请替换为实际的结果文件名
df = pd.read_csv('results/Mixtral-8x7B-v0.1_memory_benchmark_20250629_215717.csv')

# 定义颜色 - 使用论文风格的颜色
colors = {
    'shared_weights': '#5e7ce2',    # 蓝色
    'expert_weights': '#f87979',    # 深紫色
    'kv_cache': '#4ecbc4',          # 青色
    'activations': '#fdd475',       # 黄色
    'other': '#a4a4a4',             # 灰色
    'other_actual': '#f87979'       # 红色
}

# 定义场景映射
scenarios = {
    'short_input_short_output': 'Short Input / Short Output',
    'short_input_long_output': 'Short Input / Long Output',
    'long_input_short_output': 'Long Input / Short Output',
    'long_input_long_output': 'Long Input / Long Output'
}

# 设置图形大小
plt.figure(figsize=(12, 8))

# 为每个场景创建图表
for scenario_id, scenario_name in scenarios.items():
    # 过滤特定场景的数据
    scenario_data = df[df['scenario'] == scenario_id]
    
    # 设置图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 获取批次大小列表
    batch_sizes = scenario_data['batch_size'].unique()
    
    # 设置x轴位置
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    # 存储条形图底部位置
    theoretical_bottoms = np.zeros(len(batch_sizes))
    actual_bottoms = np.zeros(len(batch_sizes))
    
    # 绘制理论值的各部分
    # 1. 共享权重
    shared_weights_theoretical = scenario_data['shared_weights_theoretical'].values
    rects1 = ax.bar(x - width/2, shared_weights_theoretical, width, 
                    label='Shared Weights (Theoretical)', 
                    color=colors['shared_weights'], 
                    edgecolor='black', linewidth=0.5)
    theoretical_bottoms += shared_weights_theoretical
    
    # 2. 专家权重
    expert_weights_theoretical = scenario_data['expert_weights_theoretical'].values
    rects2 = ax.bar(x - width/2, expert_weights_theoretical, width, 
                    bottom=theoretical_bottoms, 
                    label='Expert Weights (Theoretical)', 
                    color=colors['expert_weights'], 
                    edgecolor='black', linewidth=0.5)
    theoretical_bottoms += expert_weights_theoretical
    
    # 3. KV缓存
    kv_cache_theoretical = scenario_data['kv_cache_theoretical'].values
    rects3 = ax.bar(x - width/2, kv_cache_theoretical, width, 
                    bottom=theoretical_bottoms, 
                    label='KV Cache (Theoretical)', 
                    color=colors['kv_cache'], 
                    edgecolor='black', linewidth=0.5)
    theoretical_bottoms += kv_cache_theoretical
    
    # 4. 激活值
    activations_theoretical = scenario_data['activations_theoretical'].values
    rects4 = ax.bar(x - width/2, activations_theoretical, width, 
                    bottom=theoretical_bottoms, 
                    label='Activations (Theoretical)', 
                    color=colors['activations'], 
                    edgecolor='black', linewidth=0.5)
    theoretical_bottoms += activations_theoretical
    
    # 5. 其余开销
    other_theoretical = scenario_data['other_theoretical'].values
    rects5 = ax.bar(x - width/2, other_theoretical, width, 
                    bottom=theoretical_bottoms, 
                    label='Other Overhead (Theoretical)', 
                    color=colors['other'], 
                    edgecolor='black', linewidth=0.5)
    
    # 绘制实测值的各部分
    # 1. 模型权重（对应理论中的共享权重+专家权重）
    model_weights_actual = scenario_data['model_weights_actual'].values
    rects6 = ax.bar(x + width/2, model_weights_actual, width, 
                    label='Model Weights (Actual)', 
                    color=colors['expert_weights'], 
                    edgecolor='black', linewidth=0.5)
    actual_bottoms += model_weights_actual
    
    # 2. KV缓存
    kv_cache_actual = scenario_data['kv_cache_actual'].values
    rects7 = ax.bar(x + width/2, kv_cache_actual, width, 
                    bottom=actual_bottoms, 
                    label='KV Cache (Actual)', 
                    color=colors['kv_cache'], 
                    edgecolor='black', linewidth=0.5)
    actual_bottoms += kv_cache_actual
    
    # 3. 其他显存（对应理论中的激活值+其余开销）
    other_actual = scenario_data['other_actual'].values
    rects8 = ax.bar(x + width/2, other_actual, width, 
                    bottom=actual_bottoms, 
                    label='Other (Actual)', 
                    color=colors['activations'], 
                    hatch='//', 
                    edgecolor='black', linewidth=0.5)
    
    # 添加图表标签
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Memory Usage (GB)', fontsize=12)
    ax.set_title(f'Memory Usage Comparison - {scenario_name}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    
    # 创建自定义图例
    legend_elements = [
        mpatches.Patch(facecolor=colors['shared_weights'], edgecolor='black', label='Shared Weights (Theoretical)'),
        mpatches.Patch(facecolor=colors['expert_weights'], edgecolor='black', label='Expert Weights (Theoretical)'),
        mpatches.Patch(facecolor=colors['kv_cache'], edgecolor='black', label='KV Cache'),
        mpatches.Patch(facecolor=colors['activations'], edgecolor='black', label='Activations (Theoretical)'),
        mpatches.Patch(facecolor=colors['other'], edgecolor='black', label='Other Overhead (Theoretical)'),
        mpatches.Patch(facecolor=colors['shared_weights'], edgecolor='black', label='Model Weights (Actual)'),
        mpatches.Patch(facecolor=colors['activations'], edgecolor='black', hatch='//', label='Other (Actual)')
    ]
    
    # 添加图例
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # 添加网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f'pic/{model_name}_{scenario_id}_memory_comparison.png', dpi=300, bbox_inches='tight')
    
    # 关闭当前图表
    plt.close()

print("All visualizations have been generated in the 'pic' folder.") 