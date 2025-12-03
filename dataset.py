"""
EchoNet-Dynamic Video Dataset Loader
비디오 파일을 로드하고 프레임 샘플링을 수행합니다.
"""

import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from typing import Tuple, Optional
import config


class EchoNetVideoDataset(Dataset):
    """
    EchoNet-Dynamic 비디오 데이터셋
    
    Args:
        video_dir: 비디오 파일이 있는 디렉토리
        filelist_path: FileName, EF 컬럼을 포함한 CSV 파일 경로
        num_frames: 샘플링할 프레임 수
        img_size: 리사이즈할 이미지 크기
        transform: 추가 transform (optional)
        split: 'train' or 'val' (optional, for train/val split)
    """
    
    def __init__(
        self,
        video_dir: Path,
        filelist_path: Path,
        num_frames: int = config.NUM_FRAMES,
        img_size: int = config.IMG_SIZE,
        transform: Optional[callable] = None,
        split: Optional[str] = None,
        train_ratio: float = config.TRAIN_RATIO,
    ):
        self.video_dir = Path(video_dir)
        self.filelist_path = Path(filelist_path)
        self.num_frames = num_frames
        self.img_size = img_size
        self.transform = transform
        
        # CSV 파일 로드
        if not self.filelist_path.exists():
            raise FileNotFoundError(f"FileList not found: {self.filelist_path}")
        
        df = pd.read_csv(self.filelist_path)
        
        # FileName과 EF 컬럼 확인
        if 'FileName' not in df.columns or 'EF' not in df.columns:
            raise ValueError("CSV must contain 'FileName' and 'EF' columns")
        
        # Train/Val/Test split
        if split is not None:
            # Split 컬럼이 있으면 사용
            if 'Split' in df.columns:
                split_mapping = {
                    'train': 'TRAIN',
                    'val': 'VAL',
                    'test': 'TEST',
                }
                if split in split_mapping:
                    split_value = split_mapping[split]
                    df = df[df['Split'] == split_value].reset_index(drop=True)
                else:
                    raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")
            else:
                # Split 컬럼이 없으면 train_ratio로 나눔
                n_total = len(df)
                n_train = int(n_total * train_ratio)
                
                if split == 'train':
                    df = df.iloc[:n_train].reset_index(drop=True)
                elif split == 'val':
                    df = df.iloc[n_train:].reset_index(drop=True)
                elif split == 'test':
                    # Test split이 없으면 val을 사용
                    df = df.iloc[n_train:].reset_index(drop=True)
                else:
                    raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")
        
        self.df = df.reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} samples (split: {split or 'all'})")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        비디오와 EF 레이블을 반환합니다.
        
        Returns:
            video_tensor: (num_frames, 3, H, W) shape의 텐서
            ef_label: EF 값 (float)
        """
        row = self.df.iloc[idx]
        video_filename = row['FileName']
        ef_label = float(row['EF'])
        
        # 파일명에 확장자가 없으면 .avi 추가
        if not video_filename.endswith('.avi'):
            video_filename = video_filename + '.avi'
        
        # 비디오 파일 경로
        video_path = self.video_dir / video_filename
        
        if not video_path.exists():
            # 확장자 없이도 시도
            video_path_no_ext = self.video_dir / row['FileName']
            if video_path_no_ext.exists():
                video_path = video_path_no_ext
            else:
                raise FileNotFoundError(
                    f"Video not found: {video_path}\n"
                    f"Also tried: {video_path_no_ext}\n"
                    f"Available files in directory: {list(self.video_dir.glob('*'))[:5]}"
                )
        
        # 비디오 로드 및 프레임 샘플링
        video_tensor = self._load_video(video_path)
        
        # Transform 적용 (optional)
        if self.transform:
            video_tensor = self.transform(video_tensor)
        
        return video_tensor, ef_label
    
    def _load_video(self, video_path: Path) -> torch.Tensor:
        """
        비디오 파일을 로드하고 균등간격으로 프레임을 샘플링합니다.
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            video_tensor: (num_frames, 3, H, W) shape의 텐서
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # 전체 프레임 수 확인
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.num_frames:
            # 프레임이 부족하면 반복 사용
            frame_indices = list(range(total_frames)) * (self.num_frames // total_frames + 1)
            frame_indices = frame_indices[:self.num_frames]
        else:
            # 균등간격 샘플링
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                # 실패 시 마지막 프레임 사용
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                ret, frame = cap.read()
            
            if ret:
                # BGR -> RGB 변환
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                # Normalize [0, 255] -> [0, 1]
                frame = frame.astype(np.float32) / 255.0
                # HWC -> CHW
                frame = np.transpose(frame, (2, 0, 1))
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from: {video_path}")
        
        # (num_frames, 3, H, W) 텐서로 변환
        video_tensor = torch.from_numpy(np.stack(frames))
        
        return video_tensor


def get_dataloaders(
    video_dir: Path,
    filelist_path: Path,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    num_frames: int = config.NUM_FRAMES,
    img_size: int = config.IMG_SIZE,
    train_ratio: float = config.TRAIN_RATIO,
):
    """
    Train/Val DataLoader를 생성합니다.
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = EchoNetVideoDataset(
        video_dir=video_dir,
        filelist_path=filelist_path,
        num_frames=num_frames,
        img_size=img_size,
        split='train',
        train_ratio=train_ratio,
    )
    
    val_dataset = EchoNetVideoDataset(
        video_dir=video_dir,
        filelist_path=filelist_path,
        num_frames=num_frames,
        img_size=img_size,
        split='val',
        train_ratio=train_ratio,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return train_loader, val_loader


def create_data_loaders(
    video_dir: Path,
    filelist_path: Path,
    num_frames: int = config.NUM_FRAMES,
    image_size: int = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
):
    """
    Split 컬럼 기반으로 Train/Val/Test DataLoader를 생성합니다.
    FileList.csv에 'Split' 컬럼이 있어야 합니다.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Split 컬럼 확인
    df = pd.read_csv(filelist_path)
    if 'Split' not in df.columns:
        # Split 컬럼이 없으면 기존 방식 사용 (train/val만)
        print("Warning: 'Split' column not found. Using train/val split only.")
        train_loader, val_loader = get_dataloaders(
            video_dir=video_dir,
            filelist_path=filelist_path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_frames=num_frames,
            img_size=image_size,
        )
        # Test loader는 val_loader를 복사 (실제로는 사용하지 않음)
        test_loader = val_loader
        return train_loader, val_loader, test_loader
    
    # Split 기반으로 데이터셋 생성
    train_dataset = EchoNetVideoDataset(
        video_dir=video_dir,
        filelist_path=filelist_path,
        num_frames=num_frames,
        img_size=image_size,
        split='train',
        train_ratio=0.8,  # Split 컬럼이 있으면 이 값은 무시됨
    )
    
    val_dataset = EchoNetVideoDataset(
        video_dir=video_dir,
        filelist_path=filelist_path,
        num_frames=num_frames,
        img_size=image_size,
        split='val',
        train_ratio=0.8,
    )
    
    # Test split이 있는지 확인
    if 'TEST' in df['Split'].values:
        test_dataset = EchoNetVideoDataset(
            video_dir=video_dir,
            filelist_path=filelist_path,
            num_frames=num_frames,
            img_size=image_size,
            split='test',
            train_ratio=0.8,
        )
    else:
        # Test split이 없으면 val을 test로 사용
        print("Warning: 'TEST' split not found. Using VAL split as TEST.")
        test_dataset = val_dataset
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return train_loader, val_loader, test_loader

