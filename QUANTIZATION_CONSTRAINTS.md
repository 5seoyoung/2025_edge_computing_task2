# Quantization 기술적 제약사항 및 해결

## GPU 서버 실험 결과

### Baseline (FP32) - GPU 실행
- **MAE**: 47.4066
- **Model Size**: 42.7139 MB
- **Latency**: 7.5655 ms/video (GPU 사용으로 매우 빠름)

### PTQ (INT8 - Dynamic Quantization) - CPU 실행
- **MAE**: 47.4135 (0.01% 증가, 거의 동일 ✅)
- **Model Size**: 42.7115 MB (0.01% 감소)
- **Latency**: 871.6247 ms/video (CPU 실행)

### QAT (INT8 - Dynamic Quantization) - CPU 실행
- **MAE**: 47.8161 (0.86% 증가)
- **Model Size**: 42.7115 MB (0.01% 감소)
- **Latency**: 895.1050 ms/video (CPU 실행)

## 발견된 문제 및 해결

### 1. Static Quantization 실패
- **오류**: `NotImplementedError: Could not run 'quantized::conv2d.new' with arguments from the 'CPU' backend`
- **원인**: ResNet의 skip connection이 quantized tensor 연산과 호환되지 않음
- **해결**: Dynamic Quantization으로 전환

### 2. Backend 초기화 문제
- **오류**: `RuntimeError: Didn't find engine for operation quantized::linear_prepack NoQEngine`
- **원인**: Quantization backend가 제대로 초기화되지 않음
- **해결**: `torch.backends.quantized.engine = 'qnnpack'` 명시적 설정 (모델 로드 전)

## 적용한 해결책

### Dynamic Quantization 사용
1. **Linear 레이어만 quantize**: 더 안정적이고 호환성 좋음
2. **Backend 초기화**: 모델 로드 전에 `qnnpack` backend 명시적 설정
3. **자동 CPU 이동**: Quantization 전에 모델을 CPU로 자동 이동

### QAT Fine-tuning 후 Dynamic Quantization
- QAT fine-tuning 완료 후 Dynamic Quantization 적용
- Fine-tuning된 weights를 활용하여 정확도 유지

## 제한사항

### 1. Linear 레이어만 quantize
- **결과**: Model Size 변화가 미미 (0.01% 감소)
- **이유**: ResNet backbone (Conv2d)은 quantize하지 않음
- **향후 개선**: Conv2d 레이어도 quantize하여 모델 크기 더 감소

### 2. CPU 실행
- **결과**: Latency 증가 (GPU 7.57 ms → CPU 871-895 ms)
- **이유**: Dynamic Quantization은 CPU에서만 작동
- **향후 개선**: GPU quantization 지원 (TensorRT 등)

### 3. Static Quantization 미지원
- **이유**: ResNet skip connection 호환성 문제
- **향후 개선**: 모델 구조 수정 또는 다른 quantization 라이브러리 사용

## 보고서 작성 방향

### 1. 성공한 부분
- ✅ Baseline 모델 구축 및 학습 완료 (GPU 서버)
- ✅ PTQ와 QAT Dynamic Quantization 성공적으로 적용
- ✅ 정확도 유지 (PTQ 거의 동일, QAT 약간 증가)
- ✅ 평가 시스템 구축 완료

### 2. 기술적 제약사항
- Static Quantization은 ResNet 구조로 인해 실패
- Dynamic Quantization은 Linear 레이어만 지원
- CPU 실행으로 인한 Latency 증가

### 3. 향후 연구 방향
- GPU quantization 지원 (TensorRT, ONNX Runtime)
- Conv2d 레이어도 quantize하여 모델 크기 더 감소
- Quantization-friendly 아키텍처 (MobileNet, EfficientNet)
- Custom quantization operator 구현
