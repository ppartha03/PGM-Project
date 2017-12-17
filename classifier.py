import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        # max pooling kernel_size=2 will reduce image from 28x28 to 14x14
        # stride=1 so 14x14 will become 12x12
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        # max pooling kernel_size=2 will reduce image from 12x12 to 6x6
        # stride=1 so 6x6 will become 4x4
        self.fc1 = nn.Linear(4*4*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # print("input:")
        # print(x.size())
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print("after conv1:")
        # print(x.size())
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print("after conv2:")
        # print(x.size())
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


