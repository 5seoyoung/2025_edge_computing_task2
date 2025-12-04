# Quantization 기술적 제약사항 분석

## 발견된 문제

### 1. Static Quantization 실패
- **오류**: `NotImplementedError: Could not run 'quantized::conv2d.new' with arguments from the 'CPU' backend`
- **원인**: ResNet의 skip connection이 quantized tensor 연산과 호환되지 않음
- **시도한 해결책**: 
  - Backbone만 quantize, 나머지 레이어는 FP32 유지
  - qnnpack backend 명시적 설정
  - Per-tensor quantization scheme 사용
  - 모두 실패

### 2. Dynamic Quantization 실패
- **오류**: `RuntimeError: Didn't find engine for operation quantized::linear_prepack NoQEngine`
- **원인**: Quantization backend가 제대로 초기화되지 않음
- **시도한 해결책**:
  - `torch.backends.quantized.engine = 'qnnpack'` 설정
  - 모델을 CPU로 이동
  - 모두 실패

## 이론적 기대값 vs 실제 결과

### Baseline (FP32)
- **MAE**: 41.11
- **Model Size**: 42.71 MB
- **Latency**: 503.75 ms/video

### 이론적 INT8 Quantization 기대값
- **Model Size**: ~10.68 MB (75% 감소)
- **MAE**: ~41.92 (약 2% 증가)
- **Latency**: ~428 ms/video (약 15% 개선)

### 실제 결과 (기술적 제약으로 인해)
- **Model Size**: 42.71 MB (변화 없음)
- **MAE**: 41.11 (변화 없음)
- **Latency**: 503.75 ms/video (변화 없음)

## 보고서 작성 방향

### 1. 성공한 부분
- Baseline 모델 구축 및 학습 완료
- Quantization 파이프라인 구현 완료
- 평가 시스템 구축 완료

### 2. 기술적 제약사항
- PyTorch의 quantization 지원이 ResNet과 같은 복잡한 아키텍처에서 제한적
- Skip connection을 포함한 residual block의 quantization은 추가적인 모델 구조 수정 필요
- Quantization backend 초기화 문제

### 3. 향후 연구 방향
- Quantization-friendly 아키텍처 (MobileNet, EfficientNet)
- 다른 quantization 라이브러리 (ONNX Runtime, TensorRT)
- Custom quantization operator 구현
- 모델 구조 수정 (skip connection 제거 또는 수정)

