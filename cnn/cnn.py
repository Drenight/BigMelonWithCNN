import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from cnn import cnn_model

model_path = 'model/model.pth'
if os.path.exists(model_path):
    print(f"The file {model_path} exists.")
    model = cnn_model.CNNModel()
    model.load_state_dict(torch.load(model_path))
else:
    print(f"The file {model_path} does not exist.")
    model = cnn_model.CNNModel()
    torch.save(model.state_dict(), model_path)

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

def main():
    directory_path = "./positive_data"
    file_names = os.listdir(directory_path)

    image_paths = []
    coordinates = []
    for file in file_names:
        f = os.path.join(directory_path, file)
        image_paths.append(f)
        # 通过split方法切割字符串
        parts = f.split('_')[-1]
        # 提取下划线后面点前面的部分
        coor = parts.split('.')[0]
        print(f)
        coordinates.append(int(coor))
    
    # print(image_paths)
    # print(coordinates)
    # exit()

    # image_paths = image_paths[:25]
    # coordinates = coordinates[:25]

    transform = transforms.Compose([
        transforms.Resize((1080, 1920)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(image_paths, coordinates, transform)

    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def count_parameters(model):
        print("Model parameters are following:")
        total = 0
        for name, p in model.named_parameters():
            if p.dim() > 1:
                print(f'{p.numel():,}\t{name}')
                total += p.numel()

        print(f'total = {total:,}')
        print()
    count_parameters(model)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 10
    for epoch in tqdm(range(num_epochs)):
        for batch_images, batch_coordinates in dataloader:
            batch_coordinates = batch_coordinates.long()

            # print(batch_images[1])
            # print(batch_coordinates[0])

            # print(batch_coordinates[1])
            # exit()

            batch_images, batch_coordinates = batch_images.to(device), batch_coordinates.to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)

            # print(batch)
            # print(outputs)
            # print(batch_coordinates)

            # exit()

            loss = criterion(outputs, batch_coordinates)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    torch.save(model.state_dict(), './model/model.pth')

    exit()


