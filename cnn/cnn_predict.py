import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from cnn_model import CNNModel

model = CNNModel()
model.load_state_dict(torch.load('../model/model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((1080, 1920)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
new_image = new_image.to(device)

with torch.no_grad():
    predicted_coordinates = model(new_image)

predicted_coordinates = predicted_coordinates.squeeze().cpu().numpy()

print("Predicted Coordinates:", predicted_coordinates)