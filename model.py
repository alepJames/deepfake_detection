import torch
import torch.nn as nn
from torchvision import models

class CNNWithLandmarkFusion(nn.Module):
    def __init__(self):
        super(CNNWithLandmarkFusion, self).__init__()

        # Base CNN: ResNet18 (excluding final FC)
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()  # Output size will be 512

        # MLP for landmark vector (468 points * 3 = 1404 features)
        self.mlp = nn.Sequential(
            nn.Linear(1404, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        # Classifier combining both feature types
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Output: 2 classes (real, fake)
        )

    def forward(self, image, landmarks):
        cnn_feat = self.cnn(image)               # [B, 512]
        landmark_feat = self.mlp(landmarks)      # [B, 128]
        fused = torch.cat([cnn_feat, landmark_feat], dim=1)  # [B, 640]
        out = self.classifier(fused)
        return out