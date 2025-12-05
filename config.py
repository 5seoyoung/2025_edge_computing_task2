"""
Configuration file for Quantization experiments
모든 경로와 하이퍼파라미터를 여기서 통일 관리
"""

import os
from pathlib import Path

# ==================== 경로 설정 ====================
_script_dir = Path(__file__).parent.absolute()
BASE_DIR = _script_dir / "sample_echonet_dynamic"
VIDEO_DIR = BASE_DIR / "Videos"
FILELIST_PATH = BASE_DIR / "FileList.csv"

# 결과 저장 경로
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# ==================== 데이터 설정 ====================
NUM_FRAMES = 8  # 샘플링할 프레임 수
IMG_SIZE = 112  # 이미지 크기 (112x112 또는 128x128)
BATCH_SIZE = 8
NUM_WORKERS = 2

# Train/Val split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

# ==================== 모델 설정 ====================
MODEL_NAME = "resnet18"
PRETRAINED = True
FEATURE_DIM = 512  # ResNet-18의 feature dimension

# ==================== 학습 설정 ====================
NUM_EPOCHS = 20  # Baseline 학습
QAT_EPOCHS = 5  # QAT fine-tuning epochs
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ==================== Quantization 설정 ====================
# PTQ Calibration
CALIBRATION_SAMPLES = 30  # Calibration에 사용할 샘플 수

# QAT 설정
QAT_LEARNING_RATE = 1e-5  # QAT는 더 작은 LR 사용

# ==================== 평가 설정 ====================
# Latency 측정
LATENCY_WARMUP = 10  # Warm-up iterations
LATENCY_ITERATIONS = 100  # 측정 반복 횟수

# ==================== 출력 설정 ====================
VERBOSE = True
SAVE_RESULTS = True

