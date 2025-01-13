import torch
from torch import nn
from PIL import Image
from torchvision.transforms.v2 import Compose, Resize, ToTensor, Normalize, ToDtype



class CNN(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
    '''Convolutional Neural Network'''
    super().__init__()

    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units*2, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units*2, out_channels=hidden_units*2, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_block_3 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units*2, out_channels=hidden_units*4, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units*4, out_channels=hidden_units*4, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=82944, out_features=output_shape)
    )

  def forward(self, x):
    x = self.conv_block_1(x)
    x = self.conv_block_2(x)
    x = self.conv_block_3(x)
    x = self.classifier(x)

    return x


idx_to_class = {0: 'Ulica 🛣',
                1: 'Budynek 🏢',
                2: 'Morze 🌊',
                3: 'Las 🌳',
                4: 'Lodowiec ❄️',
                5: 'Góra 🏔'}



def model_eval(model: torch.nn.Module, image_file: str) -> str:
    '''
    Model evaluating function
    '''

    image = Image.open(image_file).convert('RGB')

    transform = Compose([
       Resize((150, 150)),
       ToTensor(),
       Normalize(mean=[0.485, 0.456, 0.406], std=[0.299, 0.224, 0.225]),
       ToDtype(torch.float)
    ])

    image = transform(image)
    image = image.unsqueeze(0)

    with torch.inference_mode():
        logit = model(image)
        pred = torch.softmax(logit.squeeze(), dim=0).argmax(dim=0)
        label = idx_to_class.get(pred.item())

        return label
    

