"""
Quantization 실험 결과 시각화
보고서용 차트 및 그래프 생성
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import config

# 폰트 설정 (한글 지원)
try:
    # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
except:
    try:
        # Linux
        plt.rcParams['font.family'] = 'DejaVu Sans'
    except:
        # Fallback
        plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results(results_dir: Path = config.RESULTS_DIR):
    """
    저장된 실험 결과를 로드합니다.
    
    Args:
        results_dir: 결과 파일이 있는 디렉토리
        
    Returns:
        results: 실험 결과 딕셔너리
    """
    results_path = results_dir / "all_results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def create_comparison_chart(results: dict, save_path: Path = None):
    """
    모델 성능 비교 차트 생성 (MAE, Size, Latency)
    
    Args:
        results: 실험 결과 딕셔너리
        save_path: 저장 경로 (None이면 results/ 디렉토리에 저장)
    """
    if save_path is None:
        save_path = config.RESULTS_DIR / "comparison_chart.png"
    
    # 데이터 추출
    baseline = results.get('baseline', {})
    ptq = results.get('ptq', {})
    qat = results.get('qat', {})
    
    models = ['FP32 Baseline', 'PTQ', 'QAT']
    mae_values = [
        baseline.get('mae', 0),
        ptq.get('mae', 0),
        qat.get('mae', 0)
    ]
    size_values = [
        baseline.get('size_mb', 0),
        ptq.get('size_mb', 0),
        qat.get('size_mb', 0)
    ]
    latency_values = [
        baseline.get('latency_ms', 0),
        ptq.get('latency_ms', 0),
        qat.get('latency_ms', 0)
    ]
    
    # 3개의 subplot 생성
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # MAE 비교
    axes[0].bar(models, mae_values, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
    axes[0].set_title('Mean Absolute Error (MAE) Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('MAE', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(mae_values):
        axes[0].text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Model Size 비교
    axes[1].bar(models, size_values, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
    axes[1].set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Size (MB)', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(size_values):
        axes[1].text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Latency 비교
    axes[2].bar(models, latency_values, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
    axes[2].set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Latency (ms/video)', fontsize=12)
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(latency_values):
        axes[2].text(i, v + 10, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison chart saved to: {save_path}")
    plt.close()


def create_grouped_bar_chart(results: dict, save_path: Path = None):
    """
    그룹화된 막대 차트 생성 (모든 지표를 한 번에 비교)
    
    Args:
        results: 실험 결과 딕셔너리
        save_path: 저장 경로
    """
    if save_path is None:
        save_path = config.RESULTS_DIR / "grouped_comparison.png"
    
    # 데이터 추출
    baseline = results.get('baseline', {})
    ptq = results.get('ptq', {})
    qat = results.get('qat', {})
    
    models = ['FP32 Baseline', 'PTQ', 'QAT']
    
    # 정규화된 값 (비교를 위해)
    mae_values = [
        baseline.get('mae', 0),
        ptq.get('mae', 0),
        qat.get('mae', 0)
    ]
    size_values = [
        baseline.get('size_mb', 0),
        ptq.get('size_mb', 0),
        qat.get('size_mb', 0)
    ]
    latency_values = [
        baseline.get('latency_ms', 0),
        ptq.get('latency_ms', 0),
        qat.get('latency_ms', 0)
    ]
    
    # 정규화 (0-1 스케일)
    max_mae = max(mae_values)
    max_size = max(size_values)
    max_latency = max(latency_values)
    
    mae_norm = [v / max_mae for v in mae_values]
    size_norm = [v / max_size for v in size_values]
    latency_norm = [v / max_latency for v in latency_values]
    
    x = range(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar([i - width for i in x], mae_norm, width, label='MAE (normalized)', alpha=0.8)
    ax.bar(x, size_norm, width, label='Size (normalized)', alpha=0.8)
    ax.bar([i + width for i in x], latency_norm, width, label='Latency (normalized)', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Quantization Experiment Results (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Grouped comparison chart saved to: {save_path}")
    plt.close()


def create_performance_table(results: dict, save_path: Path = None):
    """
    성능 비교 테이블 시각화
    
    Args:
        results: 실험 결과 딕셔너리
        save_path: 저장 경로
    """
    if save_path is None:
        save_path = config.RESULTS_DIR / "performance_table.png"
    
    # 데이터 추출
    baseline = results.get('baseline', {})
    ptq = results.get('ptq', {})
    qat = results.get('qat', {})
    
    # 테이블 데이터 생성
    data = {
        'Model': ['FP32 Baseline', 'PTQ', 'QAT'],
        'Precision': ['FP32', 'INT8*', 'INT8*'],
        'MAE': [
            f"{baseline.get('mae', 0):.4f}",
            f"{ptq.get('mae', 0):.4f}",
            f"{qat.get('mae', 0):.4f}"
        ],
        'Size (MB)': [
            f"{baseline.get('size_mb', 0):.4f}",
            f"{ptq.get('size_mb', 0):.4f}",
            f"{qat.get('size_mb', 0):.4f}"
        ],
        'Latency (ms)': [
            f"{baseline.get('latency_ms', 0):.4f}",
            f"{ptq.get('latency_ms', 0):.4f}",
            f"{qat.get('latency_ms', 0):.4f}"
        ]
    }
    
    df = pd.DataFrame(data)
    
    # 테이블 시각화
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # 헤더 스타일
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 행 색상
    colors = ['#ecf0f1', '#bdc3c7', '#95a5a6']
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor(colors[i-1])
    
    plt.title('Quantization Experiment Results Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Performance table saved to: {save_path}")
    plt.close()


def create_improvement_chart(results: dict, save_path: Path = None):
    """
    Baseline 대비 개선도 차트 생성
    
    Args:
        results: 실험 결과 딕셔너리
        save_path: 저장 경로
    """
    if save_path is None:
        save_path = config.RESULTS_DIR / "improvement_chart.png"
    
    baseline = results.get('baseline', {})
    ptq = results.get('ptq', {})
    qat = results.get('qat', {})
    
    # Baseline 대비 변화율 계산
    baseline_mae = baseline.get('mae', 1)
    baseline_size = baseline.get('size_mb', 1)
    baseline_latency = baseline.get('latency_ms', 1)
    
    ptq_mae_change = ((ptq.get('mae', 0) - baseline_mae) / baseline_mae) * 100
    ptq_size_change = ((ptq.get('size_mb', 0) - baseline_size) / baseline_size) * 100
    ptq_latency_change = ((ptq.get('latency_ms', 0) - baseline_latency) / baseline_latency) * 100
    
    qat_mae_change = ((qat.get('mae', 0) - baseline_mae) / baseline_mae) * 100
    qat_size_change = ((qat.get('size_mb', 0) - baseline_size) / baseline_size) * 100
    qat_latency_change = ((qat.get('latency_ms', 0) - baseline_latency) / baseline_latency) * 100
    
    metrics = ['MAE', 'Size', 'Latency']
    ptq_changes = [ptq_mae_change, ptq_size_change, ptq_latency_change]
    qat_changes = [qat_mae_change, qat_size_change, qat_latency_change]
    
    x = range(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar([i - width/2 for i in x], ptq_changes, width, label='PTQ', alpha=0.8, color='#e74c3c')
    bars2 = ax.bar([i + width/2 for i in x], qat_changes, width, label='QAT', alpha=0.8, color='#2ecc71')
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Change from Baseline (%)', fontsize=12)
    ax.set_title('Performance Change from Baseline (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Improvement chart saved to: {save_path}")
    plt.close()


def create_all_visualizations(results_dir: Path = None):
    """
    모든 시각화 자료 생성
    
    Args:
        results_dir: 결과 디렉토리 (None이면 config.RESULTS_DIR 사용)
    """
    if results_dir is None:
        results_dir = config.RESULTS_DIR
    
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("Generating Visualization Charts")
    print("="*60)
    
    # 결과 로드
    try:
        results = load_results(results_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run experiments first to generate results.")
        return
    
    # 시각화 생성
    print("\n[1] Creating comparison chart...")
    create_comparison_chart(results, results_dir / "comparison_chart.png")
    
    print("\n[2] Creating grouped comparison chart...")
    create_grouped_bar_chart(results, results_dir / "grouped_comparison.png")
    
    print("\n[3] Creating performance table...")
    create_performance_table(results, results_dir / "performance_table.png")
    
    print("\n[4] Creating improvement chart...")
    create_improvement_chart(results, results_dir / "improvement_chart.png")
    
    print("\n" + "="*60)
    print("✅ All visualizations generated successfully!")
    print(f"📁 Saved to: {results_dir}")
    print("="*60)


if __name__ == "__main__":
    create_all_visualizations()

