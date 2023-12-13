import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from cnn import cnn_model

model = cnn_model.CNNModel()
model.load_state_dict(torch.load('model/model.pth'))
model.eval()

def main(file):
    transform = transforms.Compose([
        transforms.Resize((1080, 1920)),
        transforms.ToTensor(),
    ])
    # print(file)

    # 加载新的图片
    # new_image = Image.open('../original_data/1701557521/1.png').convert('RGB')
    new_image = Image.open(file).convert('RGB')
    new_image = transform(new_image)
    new_image = new_image.unsqueeze(0)  # 增加 batch 维度

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_image = new_image.to(device)

    with torch.no_grad():
        predicted_coordinates = model(new_image)

    # predicted_coordinates = predicted_coordinates.squeeze().cpu().numpy()
    max_values, max_indices = torch.max(predicted_coordinates, dim=-1)

    print("Predicted Coordinates:", predicted_coordinates)
    # print(max_values, max_indices)
    print(max_indices.tolist()[0])
    return max_indices.tolist()[0]