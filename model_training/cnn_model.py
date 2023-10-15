import torch.nn as nn
import torch.nn.functional as Functional


class ConvNet(nn.Module):
    def __init__(self, channel: int, size_after_pooling: int) -> None:
        self._size_after_pooling = size_after_pooling
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(channel, 6, (3, 3), (1, 1), (1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (3, 3), (1, 1), (1, 1))
        self.fc1 = nn.Linear(self._size_after_pooling, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(Functional.relu(self.conv1(x)))
        x = self.pool(Functional.relu(self.conv2(x)))
        # Flatten the layer before connecting
        # Error caused by the tensor size drop from
        # 16*32*32 = 16384
        # 16384 / 4 / 4 = 1024
        x = x.view(-1, self._size_after_pooling)
        x = Functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
