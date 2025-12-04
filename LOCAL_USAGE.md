# 로컬에서 실행하기 (Cursor/Mac)

로컬 환경에서 Quantization 실험을 실행하는 방법입니다.

## 데이터 구조 확인

데이터는 다음 위치에 있어야 합니다:
```
2025_edge_computing_task2/
├── sample_echonet_dynamic/
│   ├── Videos/
│   │   └── *.avi 파일들
│   └── FileList.csv
├── config.py
├── run_all.py
└── ...
```

## 실행 방법

### 방법 1: 스크립트 사용 (가장 간단)

```bash
# 터미널에서 프로젝트 디렉토리로 이동
cd ~/Downloads/엣지컴퓨팅_과제2/2025_edge_computing_task2

# 실행
bash run_local.sh
```

### 방법 2: Python 명령어 직접 실행

```bash
# 전체 실험 실행 (Baseline 학습 포함)
python3 run_all.py --data_root ./sample_echonet_dynamic --train_baseline

# 기존 모델 사용 (Baseline 스킵)
python3 run_all.py --data_root ./sample_echonet_dynamic --no-train_baseline
```

### 방법 3: config.py 기본값 사용

`config.py`가 이미 로컬 경로로 설정되어 있으므로, 명령줄 인자 없이도 실행 가능:

```bash
# config.py의 기본 경로 사용
python3 run_all.py --train_baseline
```

## 개별 실험 실행

```bash
# PTQ만 실행
python3 main_ptq.py --data_root ./sample_echonet_dynamic

# QAT만 실행
python3 main_qat.py --data_root ./sample_echonet_dynamic
```

## 하이퍼파라미터 조정

```bash
python3 run_all.py \
    --data_root ./sample_echonet_dynamic \
    --train_baseline \
    --batch_size 16 \
    --num_epochs 30 \
    --qat_epochs 10
```

## 출력 파일

실험 완료 후 다음 파일들이 생성됩니다:

- `results/all_results.json`: 전체 결과 (JSON)
- `results/comparison_results.csv`: 비교 테이블 (CSV)
- `checkpoints/best_model.pth`: Baseline 모델 체크포인트

## 주의사항

1. **Python 버전**: Python 3.8 이상 필요
2. **GPU**: CUDA가 있으면 자동으로 GPU 사용, 없으면 CPU 사용
3. **메모리**: 배치 크기를 조정하여 메모리 사용량 제어 가능
4. **시간**: 전체 실험은 Baseline 학습 시간에 따라 다름 (약 20 에폭)

## 문제 해결

### 경로 오류
```bash
# 데이터 경로 확인
ls -la sample_echonet_dynamic/Videos/
ls -la sample_echonet_dynamic/FileList.csv
```

### 패키지 설치
```bash
pip install -r requirements.txt
```

### GPU 확인
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

