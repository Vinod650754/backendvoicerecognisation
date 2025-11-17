"""
Neural network model definitions using PyTorch
"""
import torch
import torch.nn as nn


class WakeWordModel(nn.Module):
    """Binary classifier: 0=noise, 1=wakeword"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class CommandModel(nn.Module):
    """Multi-class classifier for commands (lock, unlock, open, close)"""
    
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class VoiceBiometricModel(nn.Module):
    """Speaker verification model for voice biometrics"""
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns embedding vector"""
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # L2 normalize the embedding
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        return x
