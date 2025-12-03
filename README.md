# Quantization 비교 실험 프로젝트

EchoNet-Dynamic 데이터셋을 기반으로 한 EF 회귀 모델에 Post-Training Quantization (PTQ)와 Quantization-Aware Training (QAT)를 적용하여 성능을 비교하는 프로젝트입니다.

## 📁 프로젝트 구조

```
quantization_project/
├── config.py             # 하이퍼파라미터 / 경로 설정
├── dataset.py            # EchoNetVideoDataset
├── model.py              # EFRegressionModel (ResNet18 + frame aggregation)
├── train.py              # 학습/평가 함수
├── quant_utils.py        # PTQ/QAT 관련 함수
├── metrics.py            # MAE, latency, model size 계산 함수
├── main_ptq.py           # PTQ 실험 실행
├── main_qat.py           # QAT 실험 실행
├── run_all.py            # baseline → PTQ → QAT 전체 수행 스크립트
├── notebooks/
│   └── quantization_experiments.ipynb   # Colab용 주력 노트북
├── requirements.txt
└── README.md
```

## 🚀 빠른 시작 (Google Colab)

1. **환경 설정**
   ```python
   !pip install -r requirements.txt
   ```

2. **데이터 준비**
   - `/content/sample_echonet/` 디렉토리에 EchoNet 샘플 데이터 업로드
   - `FileList.csv` 파일 포함 (FileName, EF 컬럼 필요)

3. **노트북 실행**
   - `notebooks/quantization_experiments.ipynb` 열기
   - 셀 순서대로 실행

## 📊 실험 결과

실험 결과는 다음 항목을 비교합니다:

- **MAE (Mean Absolute Error)**: 모델 정확도
- **Model Size (MB)**: 모델 파일 크기
- **Latency (ms/video)**: 추론 속도

### 비교 테이블 형식

| Model | Precision | Size(MB) | MAE | Latency(ms) |
| ----- | --------- | -------- | --- | ----------- |
| FP32  | FP32      | X        | X   | X           |
| PTQ   | INT8      | X        | X   | X           |
| QAT   | INT8      | X        | X   | X           |

## 🔧 주요 모듈 설명

### `dataset.py`
- EchoNet 비디오 데이터셋 로더
- 균등간격 frame sampling
- 전처리 (resize, normalize)

### `model.py`
- ResNet-18 기반 EF 회귀 모델
- Temporal mean pooling으로 프레임 집계

### `quant_utils.py`
- PTQ: Post-Training Quantization 파이프라인
- QAT: Quantization-Aware Training 파이프라인

### `metrics.py`
- MAE 계산
- Latency 측정 (GPU/CPU)
- Model size 계산

## 📝 사용 방법

### 로컬 실행
```bash
# Baseline 학습
python train.py

# PTQ 실험
python main_ptq.py

# QAT 실험
python main_qat.py

# 전체 실험 (baseline → PTQ → QAT)
python run_all.py
```

### Colab 실행
`notebooks/quantization_experiments.ipynb`를 사용하세요.

## ⚙️ 설정 변경

모든 설정은 `config.py`에서 관리됩니다:
- 데이터 경로
- 하이퍼파라미터
- Quantization 설정

## 📌 참고사항

- 본 과제는 샘플 데이터셋(약 100~200개)으로 수행됩니다
- Frame sampling: 8~16 frames
- 이미지 크기: 112×112 또는 128×128

