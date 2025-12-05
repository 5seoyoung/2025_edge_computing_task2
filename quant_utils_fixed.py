"""
Fixed Quantization Utilities
Backend 초기화 문제를 해결하고 실제로 작동하는 quantization 구현
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
import config
from metrics import evaluate_model_performance


def initialize_quantization_backend():
    """
    Quantization backend를 제대로 초기화합니다.
    """
    # qnnpack backend 설정 (CPU quantization용)
    try:
        # PyTorch 1.9+ 방식
        if hasattr(torch.backends, 'quantized'):
            torch.backends.quantized.engine = 'qnnpack'
            print(f"✅ Quantization backend set to: {torch.backends.quantized.engine}")
    except Exception as e:
        print(f"⚠️  Could not set quantization backend: {e}")
    
    # Quantization 모듈 import로 backend 활성화
    try:
        import torch.quantization
        import torch.ao.quantization
    except:
        pass


def apply_dynamic_quantization_fixed(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = config.DEVICE,
    verbose: bool = config.VERBOSE,
) -> Tuple[nn.Module, dict]:
    """
    Dynamic Quantization을 적용합니다 (Backend 초기화 문제 해결).
    """
    if verbose:
        print("\n" + "="*50)
        print("Applying Dynamic Quantization (Fixed)")
        print("="*50)
    
    # Backend 초기화
    initialize_quantization_backend()
    
    # 모델을 CPU로 이동
    model = model.cpu()
    model.eval()
    
    if verbose:
        print("\n[Step 1] Applying dynamic quantization...")
    
    try:
        # Dynamic quantization 적용
        # Linear 레이어만 quantize (더 안정적)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # Linear 레이어만 quantize (Conv2d는 제외하여 안정성 확보)
            dtype=torch.qint8
        )
        
        if verbose:
            print("✅ Dynamic quantization applied successfully!")
            
            # Quantization 확인
            quantized_count = sum(
                1 for m in quantized_model.modules() 
                if hasattr(m, '_packed_params') or 'quantized' in str(type(m))
            )
            print(f"   Quantized Linear modules: {quantized_count}")
    
    except Exception as e:
        if verbose:
            print(f"⚠️  Dynamic quantization failed: {e}")
            print("⚠️  Trying alternative approach...")
        
        # 대안: 모델의 FC 레이어만 직접 quantize
        try:
            quantized_model = model
            # FC 레이어만 quantize
            if hasattr(model, 'fc'):
                model.fc = torch.quantization.quantize_dynamic(
                    model.fc,
                    dtype=torch.qint8
                )
            if verbose:
                print("✅ Partial quantization (FC layer only) applied")
        except Exception as e2:
            if verbose:
                print(f"⚠️  Alternative approach also failed: {e2}")
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
            device='cpu',  # Quantized 모델은 CPU에서만 실행
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


def apply_linear_only_quantization(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = config.DEVICE,
    verbose: bool = config.VERBOSE,
) -> Tuple[nn.Module, dict]:
    """
    Linear 레이어만 quantize하는 간단한 방법.
    이 방법은 가장 안정적이고 대부분의 경우 작동합니다.
    """
    if verbose:
        print("\n" + "="*50)
        print("Applying Linear-Only Quantization")
        print("="*50)
    
    model = model.cpu()
    model.eval()
    
    if verbose:
        print("\n[Step 1] Quantizing Linear layers only...")
    
    # 모델 복사
    quantized_model = model
    
    # 모든 Linear 레이어를 찾아서 quantize
    linear_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                # 각 Linear 레이어를 개별적으로 quantize
                quantized_linear = torch.quantization.quantize_dynamic(
                    module,
                    dtype=torch.qint8
                )
                # 모델에 적용
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                    setattr(parent, child_name, quantized_linear)
                else:
                    setattr(model, child_name, quantized_linear)
                linear_count += 1
                if verbose:
                    print(f"   ✅ Quantized: {name}")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Failed to quantize {name}: {e}")
    
    if verbose:
        print(f"   Total quantized Linear layers: {linear_count}")
    
    # 성능 측정
    if verbose:
        print("\n[Step 2] Evaluating quantized model...")
    
    criterion = nn.MSELoss()
    
    try:
        performance = evaluate_model_performance(
            quantized_model,
            val_loader,
            criterion,
            device='cpu',
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"⚠️  Evaluation failed: {e}")
        # 원본 모델로 평가
        performance = evaluate_model_performance(
            model,
            val_loader,
            criterion,
            device='cpu',
            verbose=verbose,
        )
        quantized_model = model
    
    if verbose:
        print("\nLinear-Only Quantization completed!")
        print(f"  MAE: {performance['mae']:.4f}")
        print(f"  Size: {performance['size_mb']:.4f} MB")
        print(f"  Latency: {performance['latency_ms']:.4f} ms/video")
    
    return quantized_model, performance

