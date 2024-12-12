import torch
from torch import nn

class LinearRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.linear_layer = nn.Linear(in_features=1, out_features=1, device='cpu')

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    return self.linear_layer(X)