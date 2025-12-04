# 서버에서 실행하기

서버 환경에서 Python 스크립트로 실험을 실행하는 방법입니다.

## 빠른 시작

### 1. 전체 실험 실행 (Baseline → PTQ → QAT)

```bash
python run_all.py --data_root /path/to/echonet_dynamic --train_baseline
```

### 2. 기존 모델 사용하여 실험 실행

```bash
python run_all.py --data_root /path/to/echonet_dynamic --no-train_baseline
```

### 3. 개별 실험 실행

```bash
# PTQ만 실행
python main_ptq.py

# QAT만 실행
python main_qat.py
```

## 명령줄 옵션

### `run_all.py` 옵션

- `--data_root PATH`: 데이터 디렉토리 경로 (Videos/와 FileList.csv 포함)
- `--train_baseline`: Baseline 모델을 처음부터 학습
- `--no-train_baseline`: 기존 체크포인트 사용 (기본값)
- `--checkpoint_dir PATH`: 체크포인트 저장/로드 디렉토리 (기본값: ./checkpoints)
- `--results_dir PATH`: 결과 저장 디렉토리 (기본값: ./results)
- `--batch_size N`: 배치 크기 (기본값: 8)
- `--num_epochs N`: Baseline 학습 에폭 수 (기본값: 20)
- `--qat_epochs N`: QAT fine-tuning 에폭 수 (기본값: 5)
- `--quiet`: 출력 최소화

## 예시

### 예시 1: 기본 실행

```bash
python run_all.py \
    --data_root /data/echonet_dynamic \
    --train_baseline
```

### 예시 2: 하이퍼파라미터 커스터마이징

```bash
python run_all.py \
    --data_root /data/echonet_dynamic \
    --train_baseline \
    --batch_size 16 \
    --num_epochs 30 \
    --qat_epochs 10 \
    --checkpoint_dir ./my_checkpoints \
    --results_dir ./my_results
```

### 예시 3: 기존 모델 사용

```bash
python run_all.py \
    --data_root /data/echonet_dynamic \
    --no-train_baseline \
    --checkpoint_dir ./checkpoints
```

## 환경 변수 사용

`config.py`를 직접 수정하거나, 환경 변수를 사용할 수 있습니다:

```bash
export ECHONET_DATA_ROOT=/path/to/echonet_dynamic
python run_all.py --data_root $ECHONET_DATA_ROOT --train_baseline
```

## 배치 스크립트 사용

`run_experiment.sh` 스크립트를 수정하여 사용:

```bash
# 스크립트 수정
vim run_experiment.sh

# 실행
bash run_experiment.sh
```

## 출력 파일

실험 완료 후 다음 파일들이 생성됩니다:

- `results/all_results.json`: 전체 결과 (JSON)
- `results/comparison_results.csv`: 비교 테이블 (CSV)
- `checkpoints/best_model.pth`: Baseline 모델 체크포인트

## 주의사항

1. **데이터 경로**: `--data_root`는 Videos/ 디렉토리와 FileList.csv가 있는 상위 디렉토리여야 합니다.
2. **GPU 사용**: CUDA가 사용 가능하면 자동으로 GPU를 사용합니다.
3. **메모리**: 배치 크기를 조정하여 메모리 사용량을 제어할 수 있습니다.

