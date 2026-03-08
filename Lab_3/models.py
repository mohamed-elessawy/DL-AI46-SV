import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SimpleMLP(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(150 * 150 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.fc_layers(x)

class CustomCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64 * 18 * 18, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_transfer_model(scenario='3-A', num_classes=6):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    if scenario == '3-A':
        for param in model.parameters():
            param.requires_grad = False
            
    elif scenario == '3-B':
        for name, param in model.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
    elif scenario == '3-C':
        for param in model.parameters():
            param.requires_grad = True

    # ADDED DROPOUT HERE TO FIX THE LOSS OVERCONFIDENCE
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model
