import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.init as init

num_classes = 20

# 定义CNN模型类
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(4, 3, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(75, 37, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(97200, 100)  # Adjusted size calculation
        self.fc2 = nn.Linear(100, num_classes)
        self.softmax = nn.Softmax(-1)
        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]:
            if hasattr(layer, 'weight'):
                init.xavier_uniform_(layer.weight)

    def forward(self, x):
        # x = torch.relu(self.conv1(x))
        # x = torch.relu(self.conv2(x))
        # x = torch.relu(self.conv3(x))
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        # x = self.pool(x)

        # 展平张量
        x = x.view(-1, 97200)  # Adjusted size calculation

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

#771840/32=24120