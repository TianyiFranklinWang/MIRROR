import torch.nn as nn
from transformers import ViTModel


class Phikon(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

    def forward(self, x):
        x = self.model(x)
        x = x.last_hidden_state[:, 0, :]
        return x
