import torch.nn as nn
from torchvision import models

def initialize_model(num_classes):
    print("ResNet18 initialization...")
    weights=models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model