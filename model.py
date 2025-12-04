"""
EF Regression Model based on ResNet-18
ResNet-18을 feature extractor로 사용하고 temporal mean pooling을 적용합니다.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import config


class EFRegressionModel(nn.Module):
    """
    EF 회귀 모델
    
    구조:
    1. ResNet-18 (pretrained) - feature extractor
    2. Temporal mean pooling
    3. Linear regression head (feature_dim -> 1)
    
    입력: (B, N, C, H, W) - B: batch, N: num_frames, C: 3, H: height, W: width
    출력: (B, 1) - EF 값
    """
    
    def __init__(
        self,
        num_frames: int = config.NUM_FRAMES,
        feature_dim: int = config.FEATURE_DIM,
        pretrained: bool = config.PRETRAINED,
    ):
        super(EFRegressionModel, self).__init__()
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        
        # ResNet-18 backbone (마지막 FC 제거)
        # torchvision 0.13+에서는 weights 파라미터 사용
        if pretrained:
            try:
                # 최신 torchvision (0.13+)
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except (AttributeError, TypeError):
                # 구버전 torchvision
                resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet18(weights=None)
        # 마지막 FC와 avgpool 제거
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # ResNet-18의 마지막 conv layer 출력 크기: 512
        
        # Global Average Pooling (각 프레임에 대해)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regression head
        self.fc = nn.Linear(feature_dim, 1)
        
        # Quantization을 위한 fuse_model 지원
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (B, N, C, H, W) shape의 입력 텐서
            
        Returns:
            ef_pred: (B, 1) shape의 EF 예측값
        """
        # Quantization stub (QAT용)
        x = self.quant(x)
        
        B, N, C, H, W = x.shape
        
        # Reshape: (B, N, C, H, W) -> (B*N, C, H, W)
        x = x.view(B * N, C, H, W)
        
        # Backbone 통과
        x = self.backbone(x)  # (B*N, feature_dim, H', W')
        
        # Global Average Pooling
        x = self.avgpool(x)  # (B*N, feature_dim, 1, 1)
        x = x.view(B * N, self.feature_dim)  # (B*N, feature_dim)
        
        # Reshape: (B*N, feature_dim) -> (B, N, feature_dim)
        x = x.view(B, N, self.feature_dim)
        
        # Temporal mean pooling
        x = x.mean(dim=1)  # (B, feature_dim)
        
        # Regression head
        ef_pred = self.fc(x)  # (B, 1)
        
        # Dequantization stub (QAT용)
        ef_pred = self.dequant(ef_pred)
        
        return ef_pred
    
    def fuse_model(self):
        """
        Quantization을 위해 모델을 fuse합니다.
        ResNet-18의 conv-bn-relu를 하나로 결합합니다.
        """
        # ResNet-18의 기본 블록들을 fuse
        for module_name, module in self.backbone.named_modules():
            if isinstance(module, nn.Sequential):
                # BasicBlock 또는 Bottleneck 구조 확인
                if len(list(module.children())) >= 2:
                    try:
                        torch.quantization.fuse_modules(
                            module,
                            [['0', '1', '2']],  # conv, bn, relu
                            inplace=True
                        )
                    except:
                        pass  # fuse 실패 시 스킵


def create_model(
    num_frames: int = config.NUM_FRAMES,
    feature_dim: int = config.FEATURE_DIM,
    pretrained: bool = config.PRETRAINED,
) -> EFRegressionModel:
    """
    모델 인스턴스를 생성합니다.
    
    Returns:
        EFRegressionModel 인스턴스
    """
    model = EFRegressionModel(
        num_frames=num_frames,
        feature_dim=feature_dim,
        pretrained=pretrained,
    )
    return model

