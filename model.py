import torch
import torch.nn as nn
from torchsummary import summary

class DistillModel(nn.Module):
    def __init__(self):
        super(DistillModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    
# model = DistillModel().cuda()
# summary(model, (3,28,28))