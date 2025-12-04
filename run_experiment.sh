#!/bin/bash
# 서버에서 실험을 실행하기 위한 스크립트 예시

# 데이터 경로 설정 (서버의 실제 경로로 수정하세요)
DATA_ROOT="/path/to/echonet_dynamic"

# 결과 저장 경로
RESULTS_DIR="./results"
CHECKPOINT_DIR="./checkpoints"

# 전체 실험 실행 (Baseline 학습 포함)
python run_all.py \
    --data_root "$DATA_ROOT" \
    --train_baseline \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --batch_size 8 \
    --num_epochs 20 \
    --qat_epochs 5

# 기존 모델 사용하여 실험 실행
# python run_all.py \
#     --data_root "$DATA_ROOT" \
#     --no-train_baseline \
#     --checkpoint_dir "$CHECKPOINT_DIR" \
#     --results_dir "$RESULTS_DIR"

# PTQ만 실행
# python main_ptq.py

# QAT만 실행
# python main_qat.py

