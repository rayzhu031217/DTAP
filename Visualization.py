"""
可视化实验结果的脚本，用于生成论文图表
所有图表均基于真实实验数据生成，无占位或示意数据。
依赖: matplotlib, numpy, pymilvus
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pymilvus import Collection, connections
import config  # 用于获取 Milvus 配置和关键词

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,          # 确保文本可编辑
    'ps.fonttype': 42
})

# 关键词定义（与 utils.py 中的 is_high_value_memory 保持一致）
OBJECT_KEYWORDS = set(config.OBJECT_CLASSES)
EVENT_KEYWORDS = {"fall", "drop", "enter", "leave", "open", "close", "collide", "stop", "move"}

def classify_memory(caption: str) -> str:
    """
    根据 caption 分类记忆为：物体、事件、新物体（新颖性）
    注意：新颖性需要跨记忆的新物体检测，这里简化：若 caption 包含新物体（不在已有物体集合中）则标记为新物体。
    为了真实统计，我们可能需要记录全局物体集合，但这里我们仅使用关键词分类，不做动态更新。
    实际应用中，可以在 value_scorer 中记录 seen_objects，并在构建时保存新颖性标志。
    此处为真实分类，我们遍历所有记忆，统计出现的新物体（未在其他记忆中出现过），但这样做不严谨。
    更准确的是，我们依据构建时的 seen_objects 来标记每条记忆是否包含新物体，但实验脚本并未存储该信息。
    作为替代，我们仅统计包含物体的记忆、包含事件的记忆，并将既包含物体又包含事件的归入混合类别。
    论文中可展示三类：物体、事件、混合。
    """
    caption_lower = caption.lower()
    has_obj = any(kw in caption_lower for kw in OBJECT_KEYWORDS)
    has_evt = any(kw in caption_lower for kw in EVENT_KEYWORDS)
    if has_obj and has_evt:
        return "Object & Event"
    elif has_obj:
        return "Object"
    elif has_evt:
        return "Event"
    else:
        return "Other"

def get_memories_from_milvus():
    """从 Milvus 获取所有存储的记忆，返回列表"""
    connections.connect(alias='default', host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    collection = Collection(config.COLLECTION_NAME)
    collection.load()
    # 获取所有记忆（假设 id 连续，或使用 query）
    # 注意：如果记忆数量很多，可以分批查询，这里简化直接取全部
    results = collection.query(expr="id >= 0", output_fields=["id", "caption", "value"])
    return results

def load_results(file_path='experiment_results.json'):
    """加载实验结果的 JSON 文件"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def plot_baseline_metrics(data):
    """绘制基线指标柱状图（真实数据）"""
    baseline = data.get('baseline', {})
    retrieval = data.get('baseline_retrieval', {})

    # 选取关键指标
    metrics = {
        'HVMP': baseline.get('HVMP', 0),
        'SFC': baseline.get('SFC', 0),
        'BE (fps)': baseline.get('BE', 0),
        'MRP': retrieval.get('MRP', 0),
        'MRE (s)': retrieval.get('MRE', 0),
        'MRR': retrieval.get('MRR', 0)
    }
    # 剔除为 0 的指标（可能未计算）
    metrics = {k: v for k, v in metrics.items() if v > 0}
    if not metrics:
        print("警告：基线指标为空，跳过基线图")
        return

    names = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color='steelblue', edgecolor='black', linewidth=1)
    ax.set_ylabel('Score / Value')
    ax.set_title('Baseline Performance')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('fig_baseline.pdf')
    plt.savefig('fig_baseline.png')
    plt.close()
    print("保存基线图: fig_baseline.pdf/png")

def plot_noise_robustness(data):
    """绘制检测噪声鲁棒性折线图（含 PRR），基于真实实验数据"""
    noise_data = data.get('noise', [])
    if not noise_data:
        print("警告：无噪声实验数据，跳过噪声图")
        return

    # 按噪声水平排序
    noise_data.sort(key=lambda x: x['noise'])
    noise_levels = [item['noise'] * 100 for item in noise_data]
    hvmp = [item['HVMP'] for item in noise_data]
    sfc = [item['SFC'] for item in noise_data]
    be = [item['BE'] for item in noise_data]

    # 计算性能保持率 (PRR)：以第一个（噪声0）为基准
    base_hvmp = hvmp[0] if hvmp else 0
    base_sfc = sfc[0] if sfc else 0
    base_be = be[0] if be else 0
    prr_hvmp = [v / base_hvmp if base_hvmp > 0 else 0 for v in hvmp]
    prr_sfc = [v / base_sfc if base_sfc > 0 else 0 for v in sfc]
    prr_be = [v / base_be if base_be > 0 else 0 for v in be]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：原始指标
    ax1.plot(noise_levels, hvmp, 'o-', label='HVMP', color='C0', linewidth=2)
    ax1.plot(noise_levels, sfc, 's-', label='SFC', color='C1', linewidth=2)
    ax1.plot(noise_levels, be, '^-', label='BE (fps)', color='C2', linewidth=2)
    ax1.set_xlabel('Detection Noise Level (%)')
    ax1.set_ylabel('Score')
    ax1.set_title('Robustness to Detection Noise')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 右图：性能保持率
    ax2.plot(noise_levels, prr_hvmp, 'o-', label='PRR (HVMP)', color='C0', linewidth=2)
    ax2.plot(noise_levels, prr_sfc, 's-', label='PRR (SFC)', color='C1', linewidth=2)
    ax2.plot(noise_levels, prr_be, '^-', label='PRR (BE)', color='C2', linewidth=2)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Detection Noise Level (%)')
    ax2.set_ylabel('Performance Retention Rate')
    ax2.set_title('Performance Retention Rate (PRR)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('fig_noise_robustness.pdf')
    plt.savefig('fig_noise_robustness.png')
    plt.close()
    print("保存噪声鲁棒性图: fig_noise_robustness.pdf/png")

def plot_duration_ablation(data):
    """绘制片段时长消融实验，基于真实数据"""
    dur_data = data.get('duration', [])
    if not dur_data:
        print("警告：无片段时长数据，跳过")
        return

    dur_data.sort(key=lambda x: x['duration'])
    durations = [item['duration'] for item in dur_data]
    hvmp = [item['HVMP'] for item in dur_data]
    sfc = [item['SFC'] for item in dur_data]
    be = [item['BE'] for item in dur_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(durations, hvmp, 'o-', label='HVMP', linewidth=2, markersize=8)
    ax.plot(durations, sfc, 's-', label='SFC', linewidth=2, markersize=8)
    ax.plot(durations, be, '^-', label='BE (fps)', linewidth=2, markersize=8)
    ax.set_xlabel('Clip Duration (s)')
    ax.set_ylabel('Score')
    ax.set_title('Effect of Clip Duration')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(durations)

    plt.tight_layout()
    plt.savefig('fig_duration_ablation.pdf')
    plt.savefig('fig_duration_ablation.png')
    plt.close()
    print("保存片段时长消融图: fig_duration_ablation.pdf/png")

def plot_threshold_ablation(data):
    """绘制价值阈值消融实验，基于真实数据"""
    thr_data = data.get('threshold', [])
    if not thr_data:
        print("警告：无阈值数据，跳过")
        return

    thr_data.sort(key=lambda x: x['threshold'])
    thresholds = [item['threshold'] for item in thr_data]
    hvmp = [item['HVMP'] for item in thr_data]
    sfc = [item['SFC'] for item in thr_data]
    be = [item['BE'] for item in thr_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, hvmp, 'o-', label='HVMP', linewidth=2, markersize=8)
    ax.plot(thresholds, sfc, 's-', label='SFC', linewidth=2, markersize=8)
    ax.plot(thresholds, be, '^-', label='BE (fps)', linewidth=2, markersize=8)
    ax.set_xlabel('Value Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Effect of Value Threshold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(thresholds)

    plt.tight_layout()
    plt.savefig('fig_threshold_ablation.pdf')
    plt.savefig('fig_threshold_ablation.png')
    plt.close()
    print("保存阈值消融图: fig_threshold_ablation.pdf/png")

def plot_hvmp_pie():
    """从 Milvus 读取真实记忆，生成高价值记忆类别分布饼图"""
    try:
        memories = get_memories_from_milvus()
    except Exception as e:
        print(f"从 Milvus 读取记忆失败: {e}，跳过饼图")
        return

    if not memories:
        print("Milvus 中无记忆，跳过饼图")
        return

    # 统计各类别数量
    categories = {}
    for mem in memories:
        cat = classify_memory(mem['caption'])
        categories[cat] = categories.get(cat, 0) + 1

    # 定义顺序和颜色
    order = ['Object', 'Event', 'Object & Event', 'Other']
    labels = []
    sizes = []
    for cat in order:
        if cat in categories:
            labels.append(cat)
            sizes.append(categories[cat])
    # 添加可能未列出的其他类别
    for cat, count in categories.items():
        if cat not in order:
            labels.append(cat)
            sizes.append(count)

    if not sizes:
        print("无有效类别，跳过饼图")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                      startangle=90, colors=['#66b3ff', '#ffcc99', '#99ff99', '#ff9999'])
    ax.set_title('Distribution of Stored Memories by Type')
    plt.tight_layout()
    plt.savefig('fig_hvmp_pie.pdf')
    plt.savefig('fig_hvmp_pie.png')
    plt.close()
    print("保存高价值记忆饼图: fig_hvmp_pie.pdf/png")

def main():
    # 加载实验数据
    try:
        data = load_results()
    except FileNotFoundError:
        print("错误：找不到 experiment_results.json，请先运行 experiment.py 生成结果文件。")
        return
    except json.JSONDecodeError:
        print("错误：experiment_results.json 格式错误。")
        return

    # 生成各个图表
    plot_baseline_metrics(data)
    plot_noise_robustness(data)
    plot_duration_ablation(data)
    plot_threshold_ablation(data)
    plot_hvmp_pie()   # 直接从 Milvus 读取真实数据

    print("\n所有图表生成完毕！")

if __name__ == '__main__':
    main()
