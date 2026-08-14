import torch
import torch.nn as nn
from torchvision import models

def get_student_model(arch="resnet50", num_classes=100, device="cuda"):
    """Instantiates a ResNet backbone as the student model."""
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feature_dim = model.fc.in_features
    elif arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        feature_dim = model.fc.in_features
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        feature_dim = model.fc.in_features
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model.fc = nn.Linear(feature_dim, num_classes)
    return model.to(device), feature_dim

def forward_student_features(student_model, connector_model, images):
    """Performs manual forward extraction of raw backbone features and projected embeddings."""
    x = student_model.conv1(images)
    x = student_model.bn1(x)
    x = student_model.relu(x)
    x = student_model.maxpool(x)
    x = student_model.layer1(x)
    x = student_model.layer2(x)
    x = student_model.layer3(x)
    x = student_model.layer4(x)
    x = student_model.avgpool(x)
    
    feat_raw = torch.flatten(x, 1)
    logits = student_model.fc(feat_raw)
    
    feat_proj = connector_model(feat_raw)
    feat_proj = feat_proj / feat_proj.norm(p=2, dim=-1, keepdim=True)

    return logits, feat_proj