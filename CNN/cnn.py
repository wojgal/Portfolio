import torch
from torch import nn
from PIL import Image
from torchvision.transforms.v2 import Compose, Resize, ToTensor, Normalize, ToDtype

class CNN(nn.Module):
  def __init__(self, input_shape, hidden_units, output_shape):
    super().__init__()

    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_block_3 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=50176, out_features=output_shape)
    )

  def forward(self, x):
    x = self.conv_block_1(x)
    x = self.conv_block_2(x)
    x = self.conv_block_3(x)
    x = self.classifier(x)

    return x


idx_to_class = {'0': 'Marchewka',
                '1': 'Dynia',
                '2': 'Pomidor',
                '3': 'Fasola',
                '4': 'Papaja',
                '5': 'Kapusta',
                '6': 'Kalafior',
                '7': 'Tykwa Pospolita',
                '8': 'Przepękla Ogórkowata',
                '9': 'Bakłażan',
                '10': 'Rzodkiewka',
                '11': 'Brokół',
                '12': 'Ogórek',
                '13': 'Ziemniak',
                '14': 'Papryka'}


def load_image_to_eval(image_file):
    image = Image.open(image_file).convert('RGB')

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToDtype(torch.float)
    ])

    image = transform(image)
    image = image.unsqueeze(0)

    return image


def model_eval(model, image):
    model.eval()

    with torch.inference_mode():
        logit = model(image)
        pred = torch.softmax(logit.squeeze(), dim=0).argmax(dim=0)
        label = idx_to_class.get(str(pred.item()))

        print(label)

        return label
    

