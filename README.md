# Structured / Unstructured Pruning 비교 실험

**2025년 2학기 국민대학교 엣지컴퓨팅 강의 과제 #1**

EchoNet-Dynamic 데이터셋을 활용한 좌심실 박출률(EF) 회귀 모델에 Structured와 Unstructured Pruning을 적용하여 비교 및 분석한 프로젝트입니다.

## 과제 개요

- **과제명**: Structured / Unstructured pruning 비교하기

- **목표**: Structured / Unstructured pruning을 자유로운 task에 적용해보고 비교 및 분석하기

- **데이터셋**: EchoNet-Dynamic (심초음파 영상 데이터)

- **Task**: 좌심실 박출률(Ejection Fraction, EF) 회귀 예측

## 프로젝트 개요

본 프로젝트는 EchoNet-Dynamic 데이터셋을 기반으로 한 EF (Ejection Fraction) 회귀 모델에 **Post-Training Quantization (PTQ)**와 **Quantization-Aware Training (QAT)**를 적용하여 모델 압축 및 성능을 비교 분석합니다.

### 주요 목표
- ResNet-18 기반 EF 회귀 모델에 양자화 기법 적용
- PTQ와 QAT의 성능 비교 (MAE, Model Size, Latency)
- 엣지 디바이스 배포를 위한 모델 최적화

## 프로젝트 구조

```
2025_edge_computing_task2/
├── config.py              # 하이퍼파라미터 및 경로 설정
├── dataset.py             # EchoNetVideoDataset 클래스
├── model.py               # EFRegressionModel (ResNet18 기반)
├── train.py               # 모델 학습 및 평가 함수
├── quant_utils.py         # PTQ/QAT 양자화 유틸리티
├── metrics.py             # MAE, Latency, Model Size 계산
├── main_ptq.py            # PTQ 실험 실행 스크립트
├── main_qat.py            # QAT 실험 실행 스크립트
├── run_all.py             # 전체 실험 파이프라인 (Baseline → PTQ → QAT)
├── visualize_results.py   # 결과 시각화 스크립트
├── notebooks/
│   └── quantization_experiments.ipynb  # 실험 노트북
├── requirements.txt       # 패키지 의존성
└── README.md
```

## 설치 및 실행

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

- `sample_echonet_dynamic/` 디렉토리에 EchoNet 데이터셋 준비
- `FileList.csv` 파일 포함 (FileName, EF 컬럼 필요)
- 비디오 파일은 `Videos/` 디렉토리에 위치

### 3. 실험 실행

#### 전체 실험 실행 (Baseline 학습 포함)
```bash
python run_all.py \
    --data_root ./sample_echonet_dynamic \
    --train_baseline \
    --batch_size 16
```

#### 기존 모델 사용하여 양자화만 실행
```bash
python run_all.py \
    --data_root ./sample_echonet_dynamic \
    --no-train_baseline
```

#### 개별 실험 실행
```bash
# PTQ만 실행
python main_ptq.py --data_root ./sample_echonet_dynamic

# QAT만 실행
python main_qat.py --data_root ./sample_echonet_dynamic
```

## 실험 결과

### 성능 비교

| Model | Precision | Size (MB) | MAE | Latency (ms/video) | Device |
|-------|-----------|-----------|-----|-------------------|--------|
| FP32 Baseline | FP32 | 42.71 | 47.41 | 7.57 | GPU |
| PTQ | INT8 | 42.71 | 47.41 | 871.62 | CPU |
| QAT | INT8 | 42.71 | 47.82 | 895.11 | CPU |

### 주요 관찰 사항

- **PTQ**: 정확도 거의 유지 (MAE 증가 0.01%)
- **QAT**: 약간의 정확도 감소 (MAE 증가 0.86%)하지만 양호한 수준
- **Latency**: CPU 실행으로 인해 증가 (GPU quantization 지원 시 개선 예상)
- **Model Size**: Linear 레이어만 양자화되어 크기 감소 미미 (0.01%)

## 주요 모듈 설명

### `model.py`
- ResNet-18 기반 EF 회귀 모델
- Temporal mean pooling으로 프레임 집계
- Quantization을 위한 QuantStub/DeQuantStub 포함

### `quant_utils.py`
- **PTQ**: Post-Training Quantization 파이프라인
- **QAT**: Quantization-Aware Training 파이프라인
- Dynamic Quantization fallback 메커니즘 포함

### `metrics.py`
- MAE (Mean Absolute Error) 계산
- Latency 측정 (GPU/CPU)
- Model size 계산

### `dataset.py`
- EchoNet 비디오 데이터셋 로더
- 균등간격 frame sampling
- 전처리 (resize, normalize)

## 설정 변경

모든 설정은 `config.py`에서 관리됩니다:
- 데이터 경로 및 하이퍼파라미터
- Quantization 설정
- 학습 파라미터

## 참고사항

- 본 과제는 샘플 데이터셋(약 100~200개)으로 수행됩니다
- Frame sampling: 8 frames
- 이미지 크기: 112×112
- 양자화는 Dynamic Quantization 방식으로 Linear 레이어에 적용됩니다

## 기술 스택

- PyTorch
- torchvision
- pandas
- matplotlib
- numpy
