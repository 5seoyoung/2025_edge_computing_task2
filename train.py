"""
Training and Evaluation Functions
학습 및 평가 함수를 제공합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import config
from metrics import calculate_mae
from pathlib import Path


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str = config.DEVICE,
    verbose: bool = config.VERBOSE,
) -> float:
    """
    한 에폭 학습을 수행합니다.
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 사용할 디바이스
        verbose: 진행 상황 출력 여부
        
    Returns:
        평균 손실값
    """
    model.train()
    running_loss = 0.0
    
    if verbose:
        pbar = tqdm(train_loader, desc="Training")
    else:
        pbar = train_loader
    
    for videos, ef_labels in pbar:
        videos = videos.to(device)  # (B, N, C, H, W)
        ef_labels = ef_labels.to(device).float().unsqueeze(1)  # (B, 1)
        
        # Forward pass
        optimizer.zero_grad()
        ef_pred = model(videos)  # (B, 1)
        
        # Loss 계산
        loss = criterion(ef_pred, ef_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if verbose:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = config.DEVICE,
    verbose: bool = config.VERBOSE,
) -> Tuple[float, float]:
    """
    모델을 평가합니다.
    
    Args:
        model: 평가할 모델
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        device: 사용할 디바이스
        verbose: 진행 상황 출력 여부
        
    Returns:
        (평균 손실값, MAE)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        if verbose:
            pbar = tqdm(val_loader, desc="Evaluating")
        else:
            pbar = val_loader
        
        for videos, ef_labels in pbar:
            videos = videos.to(device)  # (B, N, C, H, W)
            ef_labels = ef_labels.to(device).float().unsqueeze(1)  # (B, 1)
            
            # Forward pass
            ef_pred = model(videos)  # (B, 1)
            
            # Loss 계산
            loss = criterion(ef_pred, ef_labels)
            running_loss += loss.item()
            
            # 예측값과 레이블 저장
            all_preds.append(ef_pred.cpu())
            all_labels.append(ef_labels.cpu())
    
    # MAE 계산
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    mae = calculate_mae(all_preds, all_labels)
    
    avg_loss = running_loss / len(val_loader)
    
    if verbose:
        print(f"Validation Loss: {avg_loss:.4f}, MAE: {mae:.4f}")
    
    return avg_loss, mae


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = config.NUM_EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    device: str = config.DEVICE,
    checkpoint_dir: Path = config.CHECKPOINT_DIR,
    save_best: bool = True,
    verbose: bool = config.VERBOSE,
) -> dict:
    """
    모델을 학습합니다.
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        num_epochs: 학습 에폭 수
        learning_rate: 학습률
        weight_decay: Weight decay
        device: 사용할 디바이스
        checkpoint_dir: 체크포인트 저장 디렉토리
        save_best: 최고 성능 모델 저장 여부
        verbose: 진행 상황 출력 여부
        
    Returns:
        학습 히스토리 딕셔너리
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
    }
    
    best_mae = float('inf')
    
    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, verbose)
        
        # Evaluate
        val_loss, val_mae = evaluate(model, val_loader, criterion, device, verbose)
        
        # History 저장
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        # Best model 저장
        if save_best and val_mae < best_mae:
            best_mae = val_mae
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_loss': val_loss,
            }, checkpoint_path)
            if verbose:
                print(f"Saved best model (MAE: {best_mae:.4f}) to {checkpoint_path}")
    
    # 최종 모델 저장
    final_checkpoint_path = checkpoint_dir / "final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
    }, final_checkpoint_path)
    
    if verbose:
        print(f"\nTraining completed. Final model saved to {final_checkpoint_path}")
    
    return history


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: str = config.DEVICE,
) -> dict:
    """
    체크포인트를 로드합니다.
    
    Args:
        model: 모델 인스턴스
        checkpoint_path: 체크포인트 파일 경로
        device: 사용할 디바이스
        
    Returns:
        체크포인트 딕셔너리
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return checkpoint

