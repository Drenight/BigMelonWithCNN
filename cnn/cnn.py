import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from cnn_model import CNNModel

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

directory_path = "../original_data"
file_names = os.listdir(directory_path)

# TODO, use positive_data to train here
# Load last time model
image_files = []
for file in file_names:
    folder_path = os.path.join(directory_path, file)
    if os.path.isdir(folder_path):
        files_in_folder = os.listdir(folder_path)
        for f in files_in_folder:
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(file_path)
        # print(len(image_files))
    else:
        # Alert
        pass
print(str(len(image_files)) + " images loaded!")
# print(image_files)
# exit()

# image_files = [file for file in file_names if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_paths = [os.path.join(directory_path, file) for file in image_files]
coordinates = [15 for _ in range(len(image_paths))]

image_paths = image_paths[:25]
coordinates = coordinates[:25]

transform = transforms.Compose([
    transforms.Resize((1080, 1920)),
    transforms.ToTensor(),
])

dataset = CustomDataset(image_paths, coordinates, transform)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CNNModel()

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
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    for batch_images, batch_coordinates in dataloader:
        batch_coordinates = batch_coordinates.long()
        batch_images, batch_coordinates = batch_images.to(device), batch_coordinates.to(device)

        optimizer.zero_grad()
        outputs = model(batch_images)
        # print(outputs)
        # print(batch_coordinates)
        loss = criterion(outputs, batch_coordinates)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

torch.save(model.state_dict(), '../model/model.pth')

exit()


