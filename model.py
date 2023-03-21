import torch
import torch.nn as nn
from torchsummary import summary

class DistillModel_DepthSep(nn.Module):
    def __init__(self):
        super(DistillModel_DepthSep, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3),
            nn.Conv2d(3, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),
            nn.Conv2d(128, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.SiLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1, groups=192),
            nn.Conv2d(192, 256, kernel_size=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Dropout(0.2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pool(x)
        feature = self.flatten(x)
        x = self.fc(feature)
        return x, feature

class DistillModel(nn.Module):
    def __init__(self):
        super(DistillModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Dropout(0.2),
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
    
# model = DistillModel_DepthSep().cuda()
# summary(model, (3,28,28))

# model = DistillModel().cuda()
# summary(model, (3,28,28))