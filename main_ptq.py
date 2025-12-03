"""
Main script for Post-Training Quantization (PTQ) experiment
PTQ 실험을 실행하고 결과를 저장합니다.
"""

import torch
import json
import pandas as pd
from pathlib import Path
import config
from model import create_model
from dataset import get_dataloaders
from train import load_checkpoint, evaluate
from quant_utils import apply_ptq
from metrics import evaluate_model_performance
import torch.nn as nn


def run_ptq_experiment(
    baseline_checkpoint_path: Path = None,
    save_results: bool = config.SAVE_RESULTS,
    verbose: bool = config.VERBOSE,
) -> dict:
    """
    PTQ 실험을 실행합니다.
    
    Args:
        baseline_checkpoint_path: Baseline 모델 체크포인트 경로
        save_results: 결과 저장 여부
        verbose: 진행 상황 출력 여부
        
    Returns:
        실험 결과 딕셔너리
    """
    device = config.DEVICE
    
    if verbose:
        print("="*60)
        print("PTQ Experiment")
        print("="*60)
    
    # 1. 데이터 로더 준비
    if verbose:
        print("\n[1] Loading dataset...")
    
    train_loader, val_loader = get_dataloaders(
        video_dir=config.VIDEO_DIR,
        filelist_path=config.FILELIST_PATH,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        num_frames=config.NUM_FRAMES,
        img_size=config.IMG_SIZE,
    )
    
    # 2. Baseline 모델 로드
    if verbose:
        print("\n[2] Loading baseline model...")
    
    if baseline_checkpoint_path is None:
        baseline_checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
    
    if not baseline_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {baseline_checkpoint_path}\n"
            "Please train baseline model first."
        )
    
    model = create_model()
    checkpoint = load_checkpoint(model, baseline_checkpoint_path, device)
    
    if verbose:
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_mae' in checkpoint:
            print(f"Baseline MAE: {checkpoint['val_mae']:.4f}")
    
    # 3. Baseline 성능 측정 (참고용)
    if verbose:
        print("\n[3] Evaluating baseline (FP32) model...")
    
    criterion = nn.MSELoss()
    baseline_performance = evaluate_model_performance(
        model,
        val_loader,
        criterion,
        device=device,
        verbose=verbose,
    )
    
    # 4. PTQ 적용
    if verbose:
        print("\n[4] Applying PTQ...")
    
    quantized_model, ptq_performance = apply_ptq(
        model,
        val_loader,
        calibration_samples=config.CALIBRATION_SAMPLES,
        device=device,
        verbose=verbose,
    )
    
    # 5. 결과 정리
    results = {
        'baseline': {
            'precision': 'FP32',
            'mae': baseline_performance['mae'],
            'size_mb': baseline_performance['size_mb'],
            'latency_ms': baseline_performance['latency_ms'],
        },
        'ptq': {
            'precision': 'INT8',
            'mae': ptq_performance['mae'],
            'size_mb': ptq_performance['size_mb'],
            'latency_ms': ptq_performance['latency_ms'],
        },
    }
    
    # Accuracy drop 계산
    mae_drop = ptq_performance['mae'] - baseline_performance['mae']
    results['ptq']['mae_drop'] = mae_drop
    results['ptq']['mae_drop_percent'] = (mae_drop / baseline_performance['mae']) * 100
    
    # Size reduction 계산
    size_reduction = baseline_performance['size_mb'] - ptq_performance['size_mb']
    results['ptq']['size_reduction_mb'] = size_reduction
    results['ptq']['size_reduction_percent'] = (size_reduction / baseline_performance['size_mb']) * 100
    
    # Latency improvement 계산
    latency_improvement = baseline_performance['latency_ms'] - ptq_performance['latency_ms']
    results['ptq']['latency_improvement_ms'] = latency_improvement
    results['ptq']['latency_improvement_percent'] = (latency_improvement / baseline_performance['latency_ms']) * 100
    
    # 6. 결과 출력
    if verbose:
        print("\n" + "="*60)
        print("PTQ Experiment Results")
        print("="*60)
        
        df = pd.DataFrame([
            {
                'Model': 'Baseline (FP32)',
                'Precision': 'FP32',
                'MAE': f"{baseline_performance['mae']:.4f}",
                'Size (MB)': f"{baseline_performance['size_mb']:.4f}",
                'Latency (ms)': f"{baseline_performance['latency_ms']:.4f}",
            },
            {
                'Model': 'PTQ (INT8)',
                'Precision': 'INT8',
                'MAE': f"{ptq_performance['mae']:.4f}",
                'Size (MB)': f"{ptq_performance['size_mb']:.4f}",
                'Latency (ms)': f"{ptq_performance['latency_ms']:.4f}",
            },
        ])
        print(df.to_string(index=False))
        
        print(f"\nMAE Drop: {mae_drop:.4f} ({results['ptq']['mae_drop_percent']:.2f}%)")
        print(f"Size Reduction: {size_reduction:.4f} MB ({results['ptq']['size_reduction_percent']:.2f}%)")
        print(f"Latency Improvement: {latency_improvement:.4f} ms ({results['ptq']['latency_improvement_percent']:.2f}%)")
    
    # 7. 결과 저장
    if save_results:
        results_dir = config.RESULTS_DIR
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # JSON 저장
        json_path = results_dir / "ptq_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # CSV 저장
        csv_path = results_dir / "ptq_results.csv"
        df.to_csv(csv_path, index=False)
        
        if verbose:
            print(f"\nResults saved to:")
            print(f"  JSON: {json_path}")
            print(f"  CSV: {csv_path}")
    
    return results


if __name__ == "__main__":
    # TODO: Colab에서 실행 시 baseline_checkpoint_path를 수정하세요
    baseline_checkpoint = config.CHECKPOINT_DIR / "best_model.pth"
    
    results = run_ptq_experiment(
        baseline_checkpoint_path=baseline_checkpoint,
        save_results=True,
        verbose=True,
    )

