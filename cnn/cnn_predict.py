import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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

# 加载训练好的模型
model = CNNModel()
model.load_state_dict(torch.load('../model/model.pth'))
model.eval()

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((1080, 1920)),
    transforms.ToTensor(),
])

# 加载新的图片
new_image = Image.open('../original_data/tryres3.png').convert('RGB')
new_image = transform(new_image)
new_image = new_image.unsqueeze(0)  # 增加 batch 维度

# 将数据移到设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
new_image = new_image.to(device)

# 使用模型进行预测
with torch.no_grad():
    predicted_coordinates = model(new_image)

# 处理输出（根据实际需要修改）
predicted_coordinates = predicted_coordinates.squeeze().cpu().numpy()

# 打印或使用预测的坐标
print("Predicted Coordinates:", predicted_coordinates)