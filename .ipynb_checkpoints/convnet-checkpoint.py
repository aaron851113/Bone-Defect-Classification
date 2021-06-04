import torch
import torch.nn as nn
import torch.nn.functional as F

#https://colab.research.google.com/drive/1fMTAqLD3qMRzSRjOrByL0JBhWMsH4dIu?usp=sharing

class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 2)
        self.conv3 = nn.Conv2d(16, 32, 5, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 224, 224)
    model = ConvNet()
    out = model(dummy_input)