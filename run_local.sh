#!/bin/bash
# 로컬에서 실행하기 위한 스크립트

# 프로젝트 디렉토리로 이동
cd "$(dirname "$0")"

# 데이터 경로 (프로젝트 루트 기준)
DATA_ROOT="./sample_echonet_dynamic"

echo "=========================================="
echo "Quantization 실험 실행"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo ""

# 전체 실험 실행 (Baseline 학습 포함)
python run_all.py \
    --data_root "$DATA_ROOT" \
    --train_baseline \
    --batch_size 8 \
    --num_epochs 20 \
    --qat_epochs 5

# 기존 모델 사용하여 실행하려면:
# python run_all.py --data_root "$DATA_ROOT" --no-train_baseline

