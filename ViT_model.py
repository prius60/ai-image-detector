import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import os
from eval import *

from transformers import CLIPProcessor, CLIPModel

# Define the ViTModel class
class ViTModel(torch.nn.Module):
    def __init__(self):
        super(ViTModel, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        # for param in self.model.parameters():
        #     param.requires_grad = True  # Freezing the CLIP model parameters
        self.fc1 = torch.nn.Linear(768, 768)
        self.fc2 = torch.nn.Linear(768, 1)

    def forward(self, images_tensor):
        outputs = self.model.get_image_features(pixel_values=images_tensor)
        outputs = self.fc1(outputs)
        outputs = torch.relu(outputs)
        outputs = self.fc2(outputs)
        return outputs


