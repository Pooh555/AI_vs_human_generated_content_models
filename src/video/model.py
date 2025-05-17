import timm
import torch.nn as nn

def create_model():
    # Load a pretrained EfficientNet B0
    model = timm.create_model('efficientnet_b0', pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)  # Binary classification
    return model
