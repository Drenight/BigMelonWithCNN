import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 定义CNN模型类
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 全连接层
        self.fc1 = nn.Linear(128 * 135 * 240, 128)  # Adjusted size calculation
        self.fc2 = nn.Linear(128, 2)  # 输出2维，分别表示x和y坐标

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        # 展平张量
        x = x.view(-1, 128 * 135 * 240)  # Adjusted size calculation

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_paths, coordinates, transform=None):
        self.image_paths = image_paths
        self.coordinates = coordinates
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        coordinates = torch.tensor(self.coordinates[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, coordinates

# 假设你有一个包含图像文件路径和相应坐标的数据集

directory_path = "../original_data"
file_names = os.listdir(directory_path)
image_files = [file for file in file_names if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_paths = [os.path.join(directory_path, file) for file in image_files]
coordinates = [[173, -61] for _ in range(len(image_paths))]  # 举例，可以根据实际情况设定坐标

image_paths = image_paths[:25]
coordinates = coordinates[:25]

# print(image_paths)
# print(coordinates)
# exit()

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((1080, 1920)),
    transforms.ToTensor(),
])

# 创建数据集实例
dataset = CustomDataset(image_paths, coordinates, transform)

# 创建数据加载器
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建模型实例
model = CNNModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将模型移到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练模型
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    for batch_images, batch_coordinates in dataloader:
        # 将数据移到GPU
        batch_images, batch_coordinates = batch_images.to(device), batch_coordinates.to(device)

        # 清零梯度，进行前向传播，计算损失，进行反向传播和优化
        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_coordinates)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# 保存训练好的模型
torch.save(model.state_dict(), '../model/model.pth')

exit()


