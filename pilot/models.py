import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class SimpleCNN(nn.Module):
    """Simple CNN with 2 conv layers and 3 fully connected layers."""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 model adapted for CIFAR-10."""

    def __init__(self, num_classes=10, pretrained=False):
        super(EfficientNetB0, self).__init__()
        # Load EfficientNet-B0
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            self.model = efficientnet_b0(weights=weights)
        else:
            self.model = efficientnet_b0(weights=None)

        # Replace the classifier head to match num_classes
        # EfficientNet-B0 has 1280 features before the classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def get_model(model_type: str = "simple_cnn", **kwargs):
    """Factory function to get a model instance."""
    if model_type == "simple_cnn":
        return SimpleCNN(**kwargs)
    elif model_type == "efficientnet_b0":
        return EfficientNetB0(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
