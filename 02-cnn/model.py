import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1   = nn.ReLU()
        self.pool1   = nn.MaxPool2d(2)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2   = nn.ReLU()
        self.pool2   = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
