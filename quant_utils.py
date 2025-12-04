"""
Quantization Utilities
Post-Training Quantization (PTQ)와 Quantization-Aware Training (QAT) 함수를 제공합니다.
"""

import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import config
from metrics import evaluate_model_performance
from train import train_one_epoch, evaluate
import torch.optim as optim


def prepare_ptq_model(model: nn.Module) -> nn.Module:
    """
    PTQ를 위해 모델을 준비합니다.
    
    Args:
        model: FP32 모델
        
    Returns:
        준비된 모델
    """
    # Quantization backend 설정 (중요!)
    # CPU에서 실행 시 qnnpack backend 사용
    if not torch.cuda.is_available():
        try:
            torch.backends.quantized.engine = 'qnnpack'
        except AttributeError:
            # 구버전 PyTorch
            pass
    
    # 모델을 eval 모드로 설정
    model.eval()
    
    # 모델을 CPU로 이동 (quantization은 CPU에서만 지원)
    model = model.cpu()
    
    # Fuse model (conv-bn-relu 결합)
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    # Quantization 설정
    # qnnpack backend 사용 (per_channel 미지원, per_tensor만 사용)
    try:
        # qnnpack backend 사용
        per_tensor_qconfig = torch.quantization.get_default_qconfig('qnnpack')
    except Exception:
        # 커스텀 per_tensor qconfig 생성
        per_tensor_qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric
            )
        )
    
    # 모델의 backbone만 quantize (skip connection 문제 방지)
    # avgpool과 fc는 FP32로 유지
    if hasattr(model, 'backbone'):
        model.backbone.qconfig = per_tensor_qconfig
        # backbone 내부의 모든 모듈에 qconfig 설정
        def set_qconfig_recursive(module):
            for child in module.children():
                if len(list(child.children())) == 0:  # leaf module
                    if hasattr(child, 'qconfig'):
                        child.qconfig = per_tensor_qconfig
                else:
                    set_qconfig_recursive(child)
            if hasattr(module, 'qconfig'):
                module.qconfig = per_tensor_qconfig
        
        set_qconfig_recursive(model.backbone)
    
    # avgpool과 fc는 quantize하지 않음 (FP32 유지)
    if hasattr(model, 'avgpool'):
        model.avgpool.qconfig = None
    if hasattr(model, 'fc'):
        model.fc.qconfig = None
    
    # Prepare
    prepared_model = torch.quantization.prepare(model, inplace=False)
    
    return prepared_model


def calibrate_model(
    prepared_model: nn.Module,
    dataloader: DataLoader,
    num_samples: int = config.CALIBRATION_SAMPLES,
    device: str = config.DEVICE,
    verbose: bool = config.VERBOSE,
) -> nn.Module:
    """
    Calibration 데이터로 모델을 캘리브레이션합니다.
    
    Args:
        prepared_model: prepare된 모델
        dataloader: Calibration 데이터 로더
        num_samples: 사용할 샘플 수
        device: 사용할 디바이스
        verbose: 진행 상황 출력 여부
        
    Returns:
        캘리브레이션된 모델
    """
    prepared_model.eval()
    # Quantization은 CPU에서만 지원되므로 CPU로 이동
    prepared_model = prepared_model.cpu()
    calibration_device = 'cpu'
    
    count = 0
    
    if verbose:
        pbar = tqdm(dataloader, desc="Calibrating")
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for videos, _ in pbar:
            if count >= num_samples:
                break
            
            videos = videos.to(calibration_device)
            _ = prepared_model(videos)
            
            count += videos.size(0)
            
            if verbose:
                pbar.set_postfix({'samples': count})
    
    if verbose:
        print(f"Calibration completed with {count} samples")
    
    return prepared_model


def convert_to_int8(prepared_model: nn.Module) -> nn.Module:
    """
    준비된 모델을 INT8로 변환합니다.
    
    Args:
        prepared_model: 캘리브레이션된 모델
        
    Returns:
        INT8 모델 (또는 FP32 모델 - quantization 실패 시)
    """
    # Quantization backend 설정 확인
    if not torch.cuda.is_available():
        try:
            torch.backends.quantized.engine = 'qnnpack'
        except AttributeError:
            pass
    
    # 모델을 CPU로 이동 (quantization은 CPU에서만 지원)
    prepared_model = prepared_model.cpu()
    prepared_model.eval()
    
    try:
        quantized_model = torch.quantization.convert(prepared_model, inplace=False)
        return quantized_model
    except (RuntimeError, NotImplementedError) as e:
        # Quantization 실패 시 FP32 모델 반환 (ResNet skip connection 문제 등)
        print(f"⚠️  Quantization conversion failed: {e}")
        print("⚠️  Falling back to FP32 model (quantization not supported for this architecture)")
        # 원본 모델의 FP32 버전 반환
        # prepared_model에서 observer 제거하고 원본 모델 반환
        return prepared_model


def apply_ptq(
    model: nn.Module,
    val_loader: DataLoader,
    calibration_samples: int = config.CALIBRATION_SAMPLES,
    device: str = config.DEVICE,
    verbose: bool = config.VERBOSE,
) -> Tuple[nn.Module, dict]:
    """
    Post-Training Quantization을 적용합니다.
    
    Pipeline:
    1. FP32 모델 로드
    2. model.eval()
    3. prepare(model)
    4. calibration (val set 일부)
    5. convert(model) -> INT8
    6. 성능 측정
    
    Args:
        model: FP32 모델
        val_loader: 검증 데이터 로더 (calibration용)
        calibration_samples: Calibration에 사용할 샘플 수
        device: 사용할 디바이스
        verbose: 진행 상황 출력 여부
        
    Returns:
        (INT8 모델, 성능 딕셔너리)
    """
    if verbose:
        print("\n" + "="*50)
        print("Applying Post-Training Quantization (PTQ)")
        print("="*50)
    
    # 1. 모델 준비
    if verbose:
        print("\n[Step 1] Preparing model for quantization...")
    prepared_model = prepare_ptq_model(model)
    
    # 2. Calibration
    if verbose:
        print(f"\n[Step 2] Calibrating with {calibration_samples} samples...")
    calibrated_model = calibrate_model(
        prepared_model,
        val_loader,
        num_samples=calibration_samples,
        device=device,
        verbose=verbose,
    )
    
    # 3. Convert to INT8
    if verbose:
        print("\n[Step 3] Converting to INT8...")
    quantized_model = convert_to_int8(calibrated_model)
    
    # 4. 성능 측정
    if verbose:
        print("\n[Step 4] Evaluating quantized model...")
    
    # INT8 모델은 CPU에서만 실행 가능
    eval_device = 'cpu'
    quantized_model = quantized_model.to(eval_device)
    
    criterion = nn.MSELoss()
    performance = evaluate_model_performance(
        quantized_model,
        val_loader,
        criterion,
        device=eval_device,
        verbose=verbose,
    )
    
    if verbose:
        print("\nPTQ completed!")
        print(f"  MAE: {performance['mae']:.4f}")
        print(f"  Size: {performance['size_mb']:.4f} MB")
        print(f"  Latency: {performance['latency_ms']:.4f} ms/video")
    
    return quantized_model, performance


def prepare_qat_model(model: nn.Module) -> nn.Module:
    """
    QAT를 위해 모델을 준비합니다.
    
    Args:
        model: FP32 모델
        
    Returns:
        준비된 모델
    """
    # Quantization backend 설정 (중요!)
    # CPU에서 실행 시 qnnpack backend 사용
    if not torch.cuda.is_available():
        try:
            torch.backends.quantized.engine = 'qnnpack'
        except AttributeError:
            # 구버전 PyTorch
            pass
    
    # 모델을 train 모드로 설정 (QAT는 학습 필요)
    model.train()
    
    # 모델을 CPU로 이동 (quantization은 CPU에서만 지원)
    # QAT는 학습 중이므로 GPU 사용 가능하지만, convert 시 CPU 필요
    # 여기서는 일단 CPU로 이동
    model = model.cpu()
    
    # Fuse model
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    # Quantization 설정
    # qnnpack backend 사용 (per_channel 미지원, per_tensor만 사용)
    try:
        # qnnpack backend 사용
        per_tensor_qat_qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    except Exception:
        # 커스텀 per_tensor qat qconfig 생성
        per_tensor_qat_qconfig = torch.quantization.QConfig(
            activation=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine
            ),
            weight=torch.quantization.FakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric
            )
        )
    
    # 모델의 backbone만 quantize (skip connection 문제 방지)
    # avgpool과 fc는 FP32로 유지
    if hasattr(model, 'backbone'):
        model.backbone.qconfig = per_tensor_qat_qconfig
        # backbone 내부의 모든 모듈에 qconfig 설정
        def set_qconfig_recursive(module):
            for child in module.children():
                if len(list(child.children())) == 0:  # leaf module
                    if hasattr(child, 'qconfig'):
                        child.qconfig = per_tensor_qat_qconfig
                else:
                    set_qconfig_recursive(child)
            if hasattr(module, 'qconfig'):
                module.qconfig = per_tensor_qat_qconfig
        
        set_qconfig_recursive(model.backbone)
    
    # avgpool과 fc는 quantize하지 않음 (FP32 유지)
    if hasattr(model, 'avgpool'):
        model.avgpool.qconfig = None
    if hasattr(model, 'fc'):
        model.fc.qconfig = None
    
    # Prepare QAT
    prepared_model = torch.quantization.prepare_qat(model, inplace=False)
    
    return prepared_model


def apply_qat(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = config.QAT_EPOCHS,
    learning_rate: float = config.QAT_LEARNING_RATE,
    device: str = config.DEVICE,
    verbose: bool = config.VERBOSE,
) -> Tuple[nn.Module, dict]:
    """
    Quantization-Aware Training을 적용합니다.
    
    Pipeline:
    1. FP32 baseline 로드
    2. prepare_qat(model)
    3. Fine-tuning (3~5 epochs)
    4. convert(model) -> INT8
    5. 성능 측정
    
    Args:
        model: FP32 baseline 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        num_epochs: Fine-tuning 에폭 수
        learning_rate: 학습률
        device: 사용할 디바이스
        verbose: 진행 상황 출력 여부
        
    Returns:
        (INT8 모델, 성능 딕셔너리)
    """
    if verbose:
        print("\n" + "="*50)
        print("Applying Quantization-Aware Training (QAT)")
        print("="*50)
    
    # 1. 모델 준비
    if verbose:
        print("\n[Step 1] Preparing model for QAT...")
    prepared_model = prepare_qat_model(model)
    prepared_model = prepared_model.to(device)
    
    # 2. Fine-tuning
    if verbose:
        print(f"\n[Step 2] Fine-tuning for {num_epochs} epochs...")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(prepared_model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        if verbose:
            print(f"\nQAT Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_one_epoch(
            prepared_model,
            train_loader,
            criterion,
            optimizer,
            device,
            verbose,
        )
        
        # Evaluate (optional, for monitoring)
        if verbose:
            evaluate(prepared_model, val_loader, criterion, device, verbose=False)
    
    # 3. Convert to INT8
    if verbose:
        print("\n[Step 3] Converting to INT8...")
    prepared_model.eval()
    prepared_model = prepared_model.cpu()  # CPU로 이동
    quantized_model = convert_to_int8(prepared_model)
    
    # 4. 성능 측정
    if verbose:
        print("\n[Step 4] Evaluating quantized model...")
    
    # 모델이 실제로 quantized되었는지 확인
    is_actually_quantized = any(
        'quantized' in str(type(m)) or hasattr(m, '_packed_params')
        for m in quantized_model.modules()
    )
    
    if not is_actually_quantized:
        if verbose:
            print("⚠️  Model is still FP32 (quantization not applied)")
    
    # INT8 모델은 CPU에서만 실행 가능, FP32는 원래 device 사용
    if is_actually_quantized:
        eval_device = 'cpu'
    else:
        eval_device = device
    
    quantized_model = quantized_model.to(eval_device)
    
    performance = evaluate_model_performance(
        quantized_model,
        val_loader,
        criterion,
        device=eval_device,
        verbose=verbose,
    )
    
    if verbose:
        print("\nQAT completed!")
        print(f"  MAE: {performance['mae']:.4f}")
        print(f"  Size: {performance['size_mb']:.4f} MB")
        print(f"  Latency: {performance['latency_ms']:.4f} ms/video")
    
    return quantized_model, performance

