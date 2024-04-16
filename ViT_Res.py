import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import os
from eval import *
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel


class ViT_Res(nn.Module):
    def __init__(self, n_layers=3, n_heads=4):
        super(ViT_Res, self).__init__()
        self.ViT = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.resnet = torchvision.models.resnet50(pretrained=True)
        
        # Project both outputs to the same dimension, here chosen as 128 for example
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)
        self.ViT_fc = nn.Linear(768, 128)

        # Freeze CLIP ViT model parameters
        # Uncomment if you want to freeze ViT parameters
        for param in self.ViT.parameters():
            param.requires_grad = False

        # Using nn.ModuleList to hold multiple attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=256, num_heads=n_heads) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(256, 1)

    def forward(self, images_tensor):
        # Process input through ViT (CLIP)
        vit_outputs = self.ViT.get_image_features(images_tensor)
        vit_outputs = self.ViT_fc(vit_outputs)

        # Process input through ResNet
        res_outputs = self.resnet(images_tensor)

        # Concatenate the outputs along the feature dimension
        combined_features = torch.cat((vit_outputs, res_outputs), dim=1)
        # Prepare for attention layer processing
        # MultiheadAttention expects (L, N, E) format, where L is the sequence length (1 in this case), N is batch size
        attention_input = combined_features.unsqueeze(0)  # Add sequence length dimension

        # Apply each attention layer sequentially
        attention_out = attention_input
        for attn_layer in self.attention_layers:
            attention_out, _ = attn_layer(attention_out, attention_out, attention_out)

        # Remove sequence length dimension
        attention_out = attention_out.squeeze(0)
        output = self.fc(attention_out)
        return output

# Example usage:
# images_tensor = torch.randn(1, 3, 224, 224)  # Example image tensor
# model = ViT_Res()
# output = model(images_tensor)
# print(output.shape)
