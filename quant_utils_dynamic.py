"""
Dynamic Quantization Utilities
Dynamic Quantization은 더 간단하고 ResNet에서도 작동할 수 있습니다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
import config
from metrics import evaluate_model_performance


def apply_dynamic_quantization(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = config.DEVICE,
    verbose: bool = config.VERBOSE,
) -> Tuple[nn.Module, dict]:
    """
    Dynamic Quantization을 적용합니다.
    Dynamic Quantization은 weights만 INT8로 변환하고, activations는 runtime에 quantize합니다.
    이는 ResNet의 skip connection과도 호환됩니다.
    
    Args:
        model: FP32 모델
        val_loader: 검증 데이터 로더
        device: 사용할 디바이스
        verbose: 진행 상황 출력 여부
        
    Returns:
        (Quantized 모델, 성능 딕셔너리)
    """
    if verbose:
        print("\n" + "="*50)
        print("Applying Dynamic Quantization")
        print("="*50)
    
    # 모델을 CPU로 이동 (Dynamic quantization은 CPU에서만 지원)
    model = model.cpu()
    model.eval()
    
    if verbose:
        print("\n[Step 1] Applying dynamic quantization...")
    
    try:
        # Dynamic quantization 적용 (weights만 INT8로 변환)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},  # Linear와 Conv2d 레이어만 quantize
            dtype=torch.qint8
        )
        
        if verbose:
            print("✅ Dynamic quantization applied successfully!")
            
            # Quantization 확인
            quantized_modules = sum(
                1 for m in quantized_model.modules() 
                if hasattr(m, '_packed_params') or 'quantized' in str(type(m))
            )
            print(f"   Quantized modules: {quantized_modules}")
    
    except Exception as e:
        if verbose:
            print(f"⚠️  Dynamic quantization failed: {e}")
            print("⚠️  Using FP32 model instead")
        quantized_model = model
    
    # 성능 측정
    if verbose:
        print("\n[Step 2] Evaluating quantized model...")
    
    criterion = nn.MSELoss()
    
    try:
        performance = evaluate_model_performance(
            quantized_model,
            val_loader,
            criterion,
            device='cpu',  # Dynamic quantized 모델은 CPU에서만 실행
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"⚠️  Quantized model evaluation failed: {e}")
            print("⚠️  Falling back to FP32 model evaluation")
        # 원본 모델로 평가
        original_model = model
        performance = evaluate_model_performance(
            original_model,
            val_loader,
            criterion,
            device='cpu',
            verbose=verbose,
        )
        quantized_model = original_model
    
    if verbose:
        print("\nDynamic Quantization completed!")
        print(f"  MAE: {performance['mae']:.4f}")
        print(f"  Size: {performance['size_mb']:.4f} MB")
        print(f"  Latency: {performance['latency_ms']:.4f} ms/video")
    
    return quantized_model, performance

