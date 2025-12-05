#!/bin/bash
# GPU 서버에서 실험 실행 스크립트

# 데이터 경로 (서버의 실제 경로로 수정하세요)
DATA_ROOT="/home/work/edgetask2/data/echonet_dynamic"

# 결과 저장 경로
RESULTS_DIR="./results"
CHECKPOINT_DIR="./checkpoints"

echo "=========================================="
echo "GPU Server Experiment Runner"
echo "=========================================="

# GPU 확인
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); if torch.cuda.is_available(): print(f'GPU name: {torch.cuda.get_device_name(0)}')"

# 데이터 경로 확인
if [ ! -d "$DATA_ROOT/Videos" ]; then
    echo "⚠️  Warning: Videos directory not found at $DATA_ROOT/Videos"
    echo "Please update DATA_ROOT in this script"
    exit 1
fi

if [ ! -f "$DATA_ROOT/FileList.csv" ]; then
    echo "⚠️  Warning: FileList.csv not found at $DATA_ROOT/FileList.csv"
    echo "Please update DATA_ROOT in this script"
    exit 1
fi

echo ""
echo "Data path: $DATA_ROOT"
echo "Results will be saved to: $RESULTS_DIR"
echo "Checkpoints will be saved to: $CHECKPOINT_DIR"
echo ""

# 전체 실험 실행
echo "Starting experiments..."
echo "=========================================="

python run_all.py \
    --data_root "$DATA_ROOT" \
    --train_baseline \
    --batch_size 16 \
    --num_epochs 20 \
    --qat_epochs 5 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --results_dir "$RESULTS_DIR"

echo ""
echo "=========================================="
echo "✅ Experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "  - all_results.json"
echo "  - comparison_results.csv"
echo "  - *.png (visualization charts)"
echo ""
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "  - best_model.pth"
echo ""

