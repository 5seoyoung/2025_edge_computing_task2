"""
Metrics Calculation Functions
MAE, Latency, Model Size 계산 함수를 제공합니다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
from pathlib import Path
import config


def calculate_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Mean Absolute Error (MAE)를 계산합니다.
    
    Args:
        predictions: 예측값 텐서 (B, 1) 또는 (B,)
        targets: 실제값 텐서 (B, 1) 또는 (B,)
        
    Returns:
        MAE 값
    """
    predictions = predictions.flatten()
    targets = targets.flatten()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    return mae


def calculate_model_size(
    model: nn.Module,
    save_path: Path = None,
) -> float:
    """
    모델 크기를 MB 단위로 계산합니다.
    
    Args:
        model: 모델 인스턴스
        save_path: 임시 저장 경로 (None이면 자동 생성)
        
    Returns:
        모델 크기 (MB)
    """
    if save_path is None:
        save_path = Path("./temp_model_size.pth")
    
    # State dict 저장
    torch.save(model.state_dict(), save_path)
    
    # 파일 크기 측정 (bytes)
    size_bytes = os.path.getsize(save_path)
    
    # MB로 변환
    size_mb = size_bytes / (1024 * 1024)
    
    # 임시 파일 삭제
    if save_path.exists() and save_path.name.startswith("temp_"):
        os.remove(save_path)
    
    return size_mb


def measure_latency(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = config.DEVICE,
    warmup: int = config.LATENCY_WARMUP,
    iterations: int = config.LATENCY_ITERATIONS,
    verbose: bool = config.VERBOSE,
) -> float:
    """
    모델의 추론 Latency를 측정합니다 (ms/video).
    
    Args:
        model: 측정할 모델
        dataloader: 데이터 로더
        device: 사용할 디바이스
        warmup: Warm-up 반복 횟수
        iterations: 측정 반복 횟수
        verbose: 진행 상황 출력 여부
        
    Returns:
        평균 Latency (ms/video)
    """
    model.eval()
    model = model.to(device)
    
    # 데이터 샘플 하나 가져오기
    sample_video, _ = next(iter(dataloader))
    sample_video = sample_video.to(device)
    
    # Warm-up
    if verbose:
        print(f"Warming up ({warmup} iterations)...")
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_video)
    
    # GPU 동기화
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Latency 측정
    if verbose:
        print(f"Measuring latency ({iterations} iterations)...")
    
    latencies = []
    
    with torch.no_grad():
        for i in range(iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(sample_video)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000  # ms로 변환
            latencies.append(latency_ms)
    
    avg_latency = sum(latencies) / len(latencies)
    
    if verbose:
        print(f"Average Latency: {avg_latency:.4f} ms/video")
        print(f"  Min: {min(latencies):.4f} ms, Max: {max(latencies):.4f} ms")
    
    return avg_latency


def evaluate_model_performance(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = config.DEVICE,
    verbose: bool = config.VERBOSE,
) -> dict:
    """
    모델의 전체 성능을 평가합니다 (MAE, Size, Latency).
    
    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        device: 사용할 디바이스
        verbose: 진행 상황 출력 여부
        
    Returns:
        성능 딕셔너리 {'mae': float, 'size_mb': float, 'latency_ms': float}
    """
    model = model.to(device)
    model.eval()
    
    # MAE 계산
    if verbose:
        print("Calculating MAE...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, ef_labels in dataloader:
            videos = videos.to(device)
            ef_labels = ef_labels.to(device).float().unsqueeze(1)
            
            ef_pred = model(videos)
            
            all_preds.append(ef_pred.cpu())
            all_labels.append(ef_labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    mae = calculate_mae(all_preds, all_labels)
    
    # Model Size 계산
    if verbose:
        print("Calculating model size...")
    size_mb = calculate_model_size(model)
    
    # Latency 측정
    if verbose:
        print("Measuring latency...")
    latency_ms = measure_latency(model, dataloader, device, verbose=verbose)
    
    results = {
        'mae': mae,
        'size_mb': size_mb,
        'latency_ms': latency_ms,
    }
    
    if verbose:
        print("\n=== Performance Summary ===")
        print(f"MAE: {mae:.4f}")
        print(f"Model Size: {size_mb:.4f} MB")
        print(f"Latency: {latency_ms:.4f} ms/video")
    
    return results

