# GPU 서버 빠른 시작 가이드

## 🚀 실행 방법

### 현재 상황
- 프로젝트 경로: `/home/work/quant/2025_edge_computing_task2/`
- 현재 위치: `/home/work/quant/`

### 해결 방법

#### 방법 1: 프로젝트 디렉토리로 이동 후 실행 (권장)

```bash
# 1. 프로젝트 디렉토리로 이동
cd /home/work/quant/2025_edge_computing_task2

# 2. 데이터 경로 확인 (실제 경로로 수정 필요)
# 예: /home/work/quant/data/echonet_dynamic

# 3. 실행
python run_all.py \
    --data_root /home/work/quant/data/echonet_dynamic \
    --train_baseline \
    --batch_size 16
```

#### 방법 2: 절대 경로로 실행

```bash
# 현재 위치에서 절대 경로로 실행
python /home/work/quant/2025_edge_computing_task2/run_all.py \
    --data_root /home/work/quant/data/echonet_dynamic \
    --train_baseline \
    --batch_size 16
```

#### 방법 3: 스크립트 사용

```bash
# 1. 프로젝트 디렉토리로 이동
cd /home/work/quant/2025_edge_computing_task2

# 2. 스크립트 수정 (데이터 경로 설정)
vim run_gpu_server.sh
# DATA_ROOT="/home/work/quant/data/echonet_dynamic" 로 수정

# 3. 실행
./run_gpu_server.sh
```

## 📝 단계별 실행 예시

```bash
# 1. 프로젝트 디렉토리로 이동
cd /home/work/quant/2025_edge_computing_task2

# 2. 현재 위치 확인
pwd
# 출력: /home/work/quant/2025_edge_computing_task2

# 3. 파일 확인
ls run_all.py
# 출력: run_all.py

# 4. 데이터 경로 확인 (실제 경로로 변경 필요)
ls /home/work/quant/data/echonet_dynamic/Videos/
ls /home/work/quant/data/echonet_dynamic/FileList.csv

# 5. 실행
python run_all.py \
    --data_root /home/work/quant/data/echonet_dynamic \
    --train_baseline \
    --batch_size 16
```

## ⚠️ 주의사항

1. **데이터 경로 확인**: `--data_root`는 Videos/ 디렉토리와 FileList.csv가 있는 상위 디렉토리
2. **작업 디렉토리**: 반드시 프로젝트 디렉토리(`2025_edge_computing_task2/`)에서 실행
3. **경로 형식**: 절대 경로 사용 권장

## 🔍 문제 해결

### "No such file or directory" 오류
```bash
# 해결: 프로젝트 디렉토리로 이동
cd /home/work/quant/2025_edge_computing_task2
python run_all.py --data_root /path/to/data
```

### "Video directory not found" 오류
```bash
# 데이터 경로 확인
ls /home/work/quant/data/echonet_dynamic/Videos/
# 경로가 다르면 --data_root를 올바른 경로로 수정
```

### GPU 확인
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

