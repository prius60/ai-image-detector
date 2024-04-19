import torch
import torchvision
from tqdm import tqdm
from eval import *

class BaselineModel(torch.nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.model = torchvision.models.resnet50()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        
        # Transformations for the images
        self.transforms = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()

    def forward(self, image_tensor):
        return self.model(image_tensor)
