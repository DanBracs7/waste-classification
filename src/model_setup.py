import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models





class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Assuming input images are 224x224
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

# (Residual Block)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # first layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut
        # If we change dimensions (stride=2) or channels, we need to adapt the original x as well
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add the original input (shortcut) to the processed output
        out += self.shortcut(x) 
        out = F.relu(out)
        return out

# Convolutional Block for VGG-like architecture    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))

#  THE COMPLETE ARCHITECTURE (ResNet18) 
class CustomResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(CustomResNet, self).__init__()
        self.in_channels = 64

        # Initial part (Stem)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The 4 stages of ResNet (each has multiple blocks)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)  # Dropout before the final layer
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        # Create a sequence of residual blocks
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        # Layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Head
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)                 #dropout before the final layer
        out = self.fc(out)
        return out

class Custom_VGG(nn.Module):
    def __init__(self, num_classes):
        super(Custom_VGG, self).__init__()
        
        # Feature Extractor: 4 blocchi che raddoppiano i canali
        self.features = nn.Sequential(
            ConvBlock(3, 32),   # Output: 112x112
            ConvBlock(32, 64),  # Output: 56x56
            ConvBlock(64, 128), # Output: 28x28
            ConvBlock(128, 256) # Output: 14x14
        )
        
        # Flatten size calcolato: 256 canali * 14 * 14 pixel = 50176
        self.flatten_dim = 256 * 14 * 14
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def initialize_model(num_classes, model_name='pretrained_resnet'):
    
    if model_name == 'pretrained_resnet':
        # pretrained model (Pre-trained)
        print("Initializing ResNet18 (Transfer Learning Pretrained - ImageNet)...")
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'custom_resnet':
        # Custom resnet model (From Scratch)
        print("Initializing Custom ResNet18 (Manual Structure)...")
        # The configuration [2, 2, 2, 2] creates a ResNet18
        model = CustomResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    elif model_name == 'custom_vgg':
        #  Model Vgg-like del 
        print("Initializing Custom VGG-like model...")
        model = Custom_VGG(num_classes)    
    else:
        raise ValueError(f"Model {model_name} not recognized.")
        
    return model