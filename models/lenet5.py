import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # (B, 1, 28, 28) -> (B, 6, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding='same'),
            nn.Sigmoid(),
            # (B, 6, 28, 28) -> (B, 6, 14, 14)
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (B, 6, 14, 14) -> (B, 16, 10, 10)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            # (B, 16, 10, 10) -> (B, 16, 5, 5)
            nn.AvgPool2d(kernel_size=2, stride=2),
            # (B, 16, 10, 10) -> (B, 400)
            nn.Flatten(),
            # (B, 400) -> (B, 120)
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Sigmoid(),
            # (B, 120) -> (B, 84)
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid(),
            # (B, 84) -> (B, 10)
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        return self.model(x.reshape(-1, 1, 28, 28))