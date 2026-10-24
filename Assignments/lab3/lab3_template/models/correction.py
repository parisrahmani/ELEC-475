"""
ELEC 475: Lab 3 - Correction Model (Starter Template)
A small CNN that predicts the offset from a patch centre to the true nose.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectionNet(nn.Module):
    """
    CorrectionNet for (3, 64, 64) ImageNet-normalized patches.

    Spatial dimension arithmetic (each block halves the size with 2x2 max-pooling):
        Input        : (B, 3, 64, 64)
        Block 1      : (B, 32, 32, 32)
        Block 2      : (B, 64, 16, 16)
        Block 3      : (B, 128, 8, 8)
        Block 4      : (B, 128, 4, 4)
        Flatten      : 128 * 4 * 4 = 2,048 features
        FC1          : 2,048 -> 256
        FC2          : 256 -> 2, the normalized offset (du, dv) / HALF_PATCH
    """
    def __init__(self):
        super(CorrectionNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2=  nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)                # (B, 32, 32, 32)
        x = self.conv2(x)                # (B, 64, 16, 16)
        x = self.conv3(x)                # (B, 128, 8, 8)
        x = self.conv4(x)                # (B, 128, 4, 4)
        x = torch.flatten(x, start_dim=1)  # (B, 2048)
        x = F.relu(self.fc1(x))          # (B, 256)
        x = self.fc2(x)                  # (B, 2), no activation
        return x


class CorrectionNetModified(nn.Module):
    """
    Your modified CorrectionNet (Post-Lab Step 5). Same input (B, 3, 64, 64) and
    output (B, 2) normalized offsets as CorrectionNet.
    """
    def __init__(self):
        super(CorrectionNetModified, self).__init__()
        # ======================================================================
        # TODO [POST-LAB STEP 5]: Implement Your Modified Correction Model
        # ======================================================================
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ======================================================================
        # TODO [POST-LAB STEP 5]: Implement the Forward Pass
        # ======================================================================
        pass


if __name__ == "__main__":
    from torchinfo import summary
    for model in [CorrectionNet(), CorrectionNetModified()]:
        print(f"Testing {type(model).__name__} Architecture:")
        try:
            summary(model, input_size=(1, 3, 64, 64))
        except Exception as e:
            print(f"Model not fully implemented yet: {e}")
