"""
Run all experiments: Baseline → PTQ → QAT
전체 실험을 순차적으로 실행하고 최종 비교 결과를 생성합니다.

서버에서 실행 예시:
    python run_all.py --data_root /path/to/echonet --train_baseline
    python run_all.py --data_root /path/to/echonet --no-train_baseline  # 기존 모델 사용
"""

import torch
import json
import pandas as pd
import argparse
from pathlib import Path
import config
from model import create_model
from dataset import get_dataloaders, create_data_loaders
from train import train_model, load_checkpoint
from main_ptq import run_ptq_experiment
from main_qat import run_qat_experiment
from metrics import evaluate_model_performance
import torch.nn as nn


def run_all_experiments(
    train_baseline: bool = True,
    save_results: bool = config.SAVE_RESULTS,
    verbose: bool = config.VERBOSE,
) -> dict:
    """
    전체 실험을 순차적으로 실행합니다.
    
    Pipeline:
    1. Baseline 모델 학습 (또는 기존 체크포인트 사용)
    2. PTQ 실험
    3. QAT 실험
    4. 최종 비교 결과 생성
    
    Args:
        train_baseline: Baseline 모델을 새로 학습할지 여부
        save_results: 결과 저장 여부
        verbose: 진행 상황 출력 여부
        
    Returns:
        전체 실험 결과 딕셔너리
    """
    device = config.DEVICE
    
    if verbose:
        print("="*60)
        print("Full Quantization Comparison Experiment")
        print("="*60)
    
    # 1. 데이터 로더 준비
    if verbose:
        print("\n[Step 1] Loading dataset...")
    
    # create_data_loaders 사용 (train/val/test 모두 반환)
    train_loader, val_loader, test_loader = create_data_loaders(
        video_dir=config.VIDEO_DIR,
        filelist_path=config.FILELIST_PATH,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    
    # 2. Baseline 모델 학습 또는 로드
    baseline_checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
    
    if train_baseline or not baseline_checkpoint_path.exists():
        if verbose:
            print("\n[Step 2] Training baseline model...")
        
        model = create_model()
        history = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=config.NUM_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            device=device,
            checkpoint_dir=config.CHECKPOINT_DIR,
            save_best=True,
            verbose=verbose,
        )
    else:
        if verbose:
            print("\n[Step 2] Loading existing baseline model...")
        
        model = create_model()
        checkpoint = load_checkpoint(model, baseline_checkpoint_path, device)
        
        if verbose:
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'val_mae' in checkpoint:
                print(f"Baseline MAE: {checkpoint['val_mae']:.4f}")
    
    # 3. Baseline 성능 측정
    if verbose:
        print("\n[Step 3] Evaluating baseline (FP32) model...")
    
    criterion = nn.MSELoss()
    baseline_performance = evaluate_model_performance(
        model,
        val_loader,
        criterion,
        device=device,
        verbose=verbose,
    )
    
    baseline_results = {
        'model': 'Baseline',
        'precision': 'FP32',
        'mae': baseline_performance['mae'],
        'size_mb': baseline_performance['size_mb'],
        'latency_ms': baseline_performance['latency_ms'],
    }
    
    # 4. PTQ 실험
    if verbose:
        print("\n" + "="*60)
        print("Starting PTQ Experiment")
        print("="*60)
    
    ptq_results = run_ptq_experiment(
        baseline_checkpoint_path=baseline_checkpoint_path,
        save_results=False,  # 최종 결과에서만 저장
        verbose=verbose,
    )
    
    # 5. QAT 실험
    if verbose:
        print("\n" + "="*60)
        print("Starting QAT Experiment")
        print("="*60)
    
    qat_results = run_qat_experiment(
        baseline_checkpoint_path=baseline_checkpoint_path,
        save_results=False,  # 최종 결과에서만 저장
        verbose=verbose,
    )
    
    # 6. 최종 비교 결과 생성
    if verbose:
        print("\n" + "="*60)
        print("Final Comparison Results")
        print("="*60)
    
    all_results = {
        'baseline': baseline_results,
        'ptq': ptq_results['ptq'],
        'qat': qat_results['qat'],
    }
    
    # 비교 테이블 생성
    comparison_df = pd.DataFrame([
        {
            'Model': 'FP32 Baseline',
            'Precision': 'FP32',
            'Size (MB)': f"{baseline_results['size_mb']:.4f}",
            'MAE': f"{baseline_results['mae']:.4f}",
            'Latency (ms)': f"{baseline_results['latency_ms']:.4f}",
        },
        {
            'Model': 'PTQ',
            'Precision': 'INT8',
            'Size (MB)': f"{ptq_results['ptq']['size_mb']:.4f}",
            'MAE': f"{ptq_results['ptq']['mae']:.4f}",
            'Latency (ms)': f"{ptq_results['ptq']['latency_ms']:.4f}",
        },
        {
            'Model': 'QAT',
            'Precision': 'INT8',
            'Size (MB)': f"{qat_results['qat']['size_mb']:.4f}",
            'MAE': f"{qat_results['qat']['mae']:.4f}",
            'Latency (ms)': f"{qat_results['qat']['latency_ms']:.4f}",
        },
    ])
    
    if verbose:
        print("\nComparison Table:")
        print(comparison_df.to_string(index=False))
        
        print("\n=== Detailed Comparison ===")
        print(f"\nBaseline (FP32):")
        print(f"  MAE: {baseline_results['mae']:.4f}")
        print(f"  Size: {baseline_results['size_mb']:.4f} MB")
        print(f"  Latency: {baseline_results['latency_ms']:.4f} ms/video")
        
        print(f"\nPTQ (INT8):")
        print(f"  MAE: {ptq_results['ptq']['mae']:.4f} (Drop: {ptq_results['ptq']['mae_drop']:.4f}, {ptq_results['ptq']['mae_drop_percent']:.2f}%)")
        print(f"  Size: {ptq_results['ptq']['size_mb']:.4f} MB (Reduction: {ptq_results['ptq']['size_reduction_mb']:.4f} MB, {ptq_results['ptq']['size_reduction_percent']:.2f}%)")
        print(f"  Latency: {ptq_results['ptq']['latency_ms']:.4f} ms/video (Improvement: {ptq_results['ptq']['latency_improvement_ms']:.4f} ms, {ptq_results['ptq']['latency_improvement_percent']:.2f}%)")
        
        print(f"\nQAT (INT8):")
        print(f"  MAE: {qat_results['qat']['mae']:.4f} (Drop: {qat_results['qat']['mae_drop']:.4f}, {qat_results['qat']['mae_drop_percent']:.2f}%)")
        print(f"  Size: {qat_results['qat']['size_mb']:.4f} MB (Reduction: {qat_results['qat']['size_reduction_mb']:.4f} MB, {qat_results['qat']['size_reduction_percent']:.2f}%)")
        print(f"  Latency: {qat_results['qat']['latency_ms']:.4f} ms/video (Improvement: {qat_results['qat']['latency_improvement_ms']:.4f} ms, {qat_results['qat']['latency_improvement_percent']:.2f}%)")
        
        print(f"\nPTQ vs QAT:")
        mae_diff = ptq_results['ptq']['mae'] - qat_results['qat']['mae']
        print(f"  MAE Difference: {mae_diff:.4f} (QAT is {'better' if mae_diff > 0 else 'worse'})")
    
    # 7. 결과 저장
    if save_results:
        results_dir = config.RESULTS_DIR
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # JSON 저장
        json_path = results_dir / "all_results.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # CSV 저장
        csv_path = results_dir / "comparison_results.csv"
        comparison_df.to_csv(csv_path, index=False)
        
        if verbose:
            print(f"\nResults saved to:")
            print(f"  JSON: {json_path}")
            print(f"  CSV: {csv_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full quantization comparison experiment (Baseline → PTQ → QAT)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to data directory (should contain Videos/ and FileList.csv). "
             "If not provided, uses config.BASE_DIR"
    )
    parser.add_argument(
        "--train_baseline",
        action="store_true",
        default=False,
        help="Train baseline model from scratch (default: use existing checkpoint if available)"
    )
    parser.add_argument(
        "--no-train_baseline",
        dest="train_baseline",
        action="store_false",
        help="Skip baseline training, use existing checkpoint"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save/load checkpoints (default: ./checkpoints)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory to save results (default: ./results)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=f"Batch size (default: {config.BATCH_SIZE})"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help=f"Number of epochs for baseline training (default: {config.NUM_EPOCHS})"
    )
    parser.add_argument(
        "--qat_epochs",
        type=int,
        default=None,
        help=f"Number of epochs for QAT fine-tuning (default: {config.QAT_EPOCHS})"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # 경로 설정 업데이트
    if args.data_root:
        config.BASE_DIR = Path(args.data_root)
        config.VIDEO_DIR = config.BASE_DIR / "Videos"
        config.FILELIST_PATH = config.BASE_DIR / "FileList.csv"
        print(f"Using data from: {config.BASE_DIR}")
    
    if args.checkpoint_dir:
        config.CHECKPOINT_DIR = Path(args.checkpoint_dir)
        config.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    
    if args.results_dir:
        config.RESULTS_DIR = Path(args.results_dir)
        config.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    # 하이퍼파라미터 업데이트
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    if args.num_epochs:
        config.NUM_EPOCHS = args.num_epochs
    
    if args.qat_epochs:
        config.QAT_EPOCHS = args.qat_epochs
    
    # 데이터 경로 확인
    if not config.VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {config.VIDEO_DIR}")
    if not config.FILELIST_PATH.exists():
        raise FileNotFoundError(f"FileList.csv not found: {config.FILELIST_PATH}")
    
    # 실험 실행
    results = run_all_experiments(
        train_baseline=args.train_baseline,
        save_results=True,
        verbose=not args.quiet,
    )
    
    print("\n✅ All experiments completed successfully!")

