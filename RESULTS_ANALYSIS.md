# 실험 결과 분석

## GPU 서버 실행 결과 요약

### Baseline (FP32) - GPU에서 실행
- **MAE**: 47.4066
- **Model Size**: 42.7139 MB
- **Latency**: 7.5655 ms/video ⚡ (GPU 사용으로 매우 빠름)

### PTQ (INT8 - Dynamic Quantization) - CPU에서 실행
- **MAE**: 47.4135 (0.01% 증가, 거의 동일 ✅)
- **Model Size**: 42.7115 MB (0.01% 감소)
- **Latency**: 871.6247 ms/video (CPU 실행으로 느림)

### QAT (INT8 - Dynamic Quantization) - CPU에서 실행
- **MAE**: 47.8161 (0.86% 증가)
- **Model Size**: 42.7115 MB (0.01% 감소)
- **Latency**: 895.1050 ms/video (CPU 실행으로 느림)

## 결과 분석

### ✅ 성공한 부분

1. **Quantization 적용 성공**
   - PTQ와 QAT 모두 Dynamic Quantization이 정상 작동
   - MAE 변화 관찰 (실제 quantization 적용 확인)

2. **정확도 유지**
   - PTQ: MAE 거의 변화 없음 (0.01% 증가)
   - QAT: 약간 증가 (0.86%)하지만 여전히 양호

3. **모델 크기**
   - 약간 감소 (0.01%) - Linear 레이어만 quantize했기 때문

### ⚠️ 주의할 점

1. **Latency 증가**
   - Baseline (GPU): 7.5 ms
   - Quantized (CPU): 871-895 ms
   - **이유**: Quantized 모델이 CPU에서 실행되기 때문
   - GPU에서 실행하면 훨씬 빠를 것으로 예상

2. **Model Size 변화 미미**
   - Linear 레이어만 quantize했기 때문에 전체 크기 변화가 작음
   - ResNet backbone (Conv2d)은 quantize하지 않음

## 보고서 작성 시 강조할 점

### 1. Quantization 성공
- PTQ와 QAT 모두 정상 작동
- 실제 quantization 적용 확인 (MAE 변화 관찰)

### 2. 정확도 유지
- PTQ: 거의 변화 없음 (0.01% 증가)
- QAT: 약간 증가 (0.86%)하지만 양호

### 3. 기술적 제약
- Dynamic Quantization은 CPU에서만 작동
- GPU에서 실행 시 latency가 크게 개선될 것으로 예상
- Linear 레이어만 quantize하여 전체 크기 변화가 미미

### 4. 향후 연구 방향
- GPU quantization 지원 (TensorRT 등)
- Conv2d 레이어도 quantize하여 모델 크기 더 감소
- Static Quantization 재시도 (ResNet 구조 수정)

## 결과 비교표

| Model | Precision | MAE | Size (MB) | Latency (ms) | Device |
|-------|-----------|-----|-----------|--------------|--------|
| Baseline | FP32 | 47.41 | 42.71 | 7.57 | GPU |
| PTQ | INT8 | 47.41 | 42.71 | 871.62 | CPU |
| QAT | INT8 | 47.82 | 42.71 | 895.11 | CPU |

## 결론

- ✅ Quantization 파이프라인 정상 작동
- ✅ 정확도 유지 (PTQ 거의 동일, QAT 약간 증가)
- ⚠️ Latency는 CPU 실행으로 인해 증가 (GPU 사용 시 개선 예상)
- ⚠️ Model Size는 Linear 레이어만 quantize하여 변화 미미

