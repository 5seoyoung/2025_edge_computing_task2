# GPU 서버에서 실행하기

GPU 서버에서 실험을 실행하는 방법입니다.

## 🔍 현재 코드의 GPU/CPU 처리 방식

### 자동 디바이스 선택
- **학습/평가**: GPU가 있으면 자동으로 GPU 사용 (`config.DEVICE`가 자동으로 `"cuda"`로 설정)
- **Quantization**: Dynamic Quantization은 CPU에서만 작동하므로 자동으로 CPU로 이동
- **결과**: 학습은 빠르게, quantization은 안정적으로 작동

### 코드 동작 흐름
1. **Baseline 학습**: GPU에서 학습 (빠름)
2. **PTQ/QAT**: 
   - 모델을 CPU로 이동
   - Dynamic Quantization 적용 (CPU에서만 작동)
   - 평가는 CPU에서 수행

## 🚀 GPU 서버에서 실행하기

### 1. 전체 실험 실행 (권장)

```bash
# Baseline 학습부터 PTQ, QAT까지 모두 실행
python run_all.py \
    --data_root /path/to/echonet_dynamic \
    --train_baseline \
    --batch_size 16 \
    --num_epochs 20 \
    --qat_epochs 5
```

### 2. 기존 모델 사용하여 실험

```bash
# 이미 학습된 Baseline 모델이 있는 경우
python run_all.py \
    --data_root /path/to/echonet_dynamic \
    --no-train_baseline
```

### 3. 개별 실험 실행

```bash
# PTQ만 실행
python main_ptq.py --data_root /path/to/echonet_dynamic

# QAT만 실행
python main_qat.py --data_root /path/to/echonet_dynamic
```

## 📝 GPU 서버 실행 스크립트 예시

`run_gpu_server.sh` 파일을 생성:

```bash
#!/bin/bash
# GPU 서버에서 실험 실행 스크립트

# 데이터 경로 (서버의 실제 경로로 수정)
DATA_ROOT="/home/work/edgetask2/data/echonet_dynamic"

# GPU 확인
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 전체 실험 실행
python run_all.py \
    --data_root "$DATA_ROOT" \
    --train_baseline \
    --batch_size 16 \
    --num_epochs 20 \
    --qat_epochs 5 \
    --checkpoint_dir ./checkpoints \
    --results_dir ./results

echo "✅ Experiments completed!"
```

실행:
```bash
chmod +x run_gpu_server.sh
./run_gpu_server.sh
```

## ⚙️ GPU 서버에서의 동작

### 학습 단계 (GPU 사용)
- **Baseline 학습**: GPU에서 빠르게 학습
- **QAT Fine-tuning**: GPU에서 학습 (5 epochs)

### Quantization 단계 (CPU 사용)
- **PTQ/QAT Conversion**: 
  - 모델이 자동으로 CPU로 이동
  - Dynamic Quantization 적용 (CPU에서만 작동)
  - 평가도 CPU에서 수행

### 성능 차이
- **학습 속도**: GPU 사용 시 CPU 대비 10-50배 빠름
- **Quantization**: CPU에서 수행 (GPU 지원 없음)
- **평가 속도**: Quantized 모델은 CPU에서 평가

## 🔧 환경 설정

### 1. 가상환경 활성화 (필요시)

```bash
# conda 환경
conda activate your_env

# 또는 venv
source venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. GPU 확인

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## 📊 예상 실행 시간 (GPU 서버 기준)

- **Baseline 학습** (20 epochs): ~10-30분 (GPU 사용)
- **PTQ**: ~5-10분 (CPU quantization)
- **QAT Fine-tuning** (5 epochs): ~5-15분 (GPU 사용)
- **QAT Quantization**: ~5-10분 (CPU quantization)

**총 예상 시간**: ~30-60분

## ⚠️ 주의사항

1. **데이터 경로**: `--data_root`는 Videos/ 디렉토리와 FileList.csv가 있는 상위 디렉토리
2. **메모리**: GPU 메모리가 부족하면 `--batch_size`를 줄이세요 (예: 8 또는 4)
3. **Quantization**: Dynamic Quantization은 CPU에서만 작동하므로, quantization 단계는 CPU로 자동 이동됩니다
4. **결과**: 모든 결과는 `results/` 디렉토리에 저장됩니다

## 🐛 문제 해결

### GPU가 인식되지 않는 경우
```bash
# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# CUDA 버전 확인
nvidia-smi
```

### 메모리 부족 오류
```bash
# 배치 크기 줄이기
python run_all.py --data_root /path/to/data --batch_size 4
```

### Quantization 오류
- Dynamic Quantization은 CPU에서만 작동합니다
- 코드가 자동으로 CPU로 이동하므로 수동 조작 불필요

## 📁 출력 파일

실험 완료 후:
- `results/all_results.json`: 전체 결과
- `results/comparison_results.csv`: 비교 테이블
- `checkpoints/best_model.pth`: Baseline 모델
- `results/*.png`: 시각화 차트

## 🎯 빠른 시작 예시

```bash
# 1. 데이터 경로 확인
ls /path/to/echonet_dynamic/Videos/
ls /path/to/echonet_dynamic/FileList.csv

# 2. 전체 실험 실행
python run_all.py \
    --data_root /path/to/echonet_dynamic \
    --train_baseline

# 3. 결과 확인
cat results/comparison_results.csv
ls results/*.png
```

