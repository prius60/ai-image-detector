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

def extract_random_patch(image, min_size=3, max_size=10):
    
    # Image dimensions
    B, C, H, W = image.shape

    # Choose a random size for the patch
    patch_size = random.randint(min_size, max_size)

    # Randomly choose the top-left corner of the patch
    if H - patch_size > 0:
        top = random.randint(0, H - patch_size)
    else:
        top = 0

    if W - patch_size > 0:
        left = random.randint(0, W - patch_size)
    else:
        left = 0

    # Extract the patch
    patch = image[:, :, top:top + patch_size, left:left + patch_size]

    return patch


class ViT_Res_patch(nn.Module):
    def __init__(self, n_layers=3, n_heads=4):
        super(ViT_Res_patch, self).__init__()
        self.ViT = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.resnet = torchvision.models.resnet50(pretrained=True)
        
        # Project both outputs to the same dimension, here chosen as 128 for example
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)
        self.ViT_fc = nn.Linear(768, 128)

        # Freeze CLIP ViT model parameters
        # Uncomment if you want to freeze ViT parameters
        for param in self.ViT.parameters():
            param.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc.requires_grad = True

        # Using nn.ModuleList to hold multiple attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=128, num_heads=n_heads) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(128, 1)

    def forward(self, images_tensor):
        # Process input through ViT (CLIP)
        vit_outputs = self.ViT.get_image_features(images_tensor)
        vit_outputs = self.ViT_fc(vit_outputs)  # Add sequence length dimension

        # Process input through ResNet
        res_outputs = self.resnet(images_tensor)  # Add sequence length dimension
        

        # Extract random patches from the input image
        patchs = [extract_random_patch(images_tensor) for _ in range(6)]
        # resize the patches to 224x224
        patchs = [torch.nn.functional.interpolate(patch, size=(224, 224)) for patch in patchs]
        # pass the patches through the resnet
        embeddings = [vit_outputs, res_outputs]
        for patch in patchs:
            embeddings.append(self.resnet(patch))
        
        # Concatenate the outputs at sequence length dimension
        combined_features = torch.stack(embeddings, dim=0)
        attention_out = combined_features
        for attn_layer in self.attention_layers:
            attention_out, _ = attn_layer(attention_out, attention_out, attention_out)


        
        # Prepare for attention layer processing
        # MultiheadAttention expects (L, N, E) format, where L is the sequence length (1 in this case), N is batch size

        # Apply each attention layer sequentially
        # vit_outputs = vit_outputs.unsqueeze(0)
        # res_outputs = res_outputs.unsqueeze(0)
        # attention_out = torch.cat([vit_outputs, res_outputs], dim=0)
        # for attn_layer in self.attention_layers:
        #     attention_out, _ = attn_layer(attention_out, attention_out, attention_out)

        
        # Apply final linear layer
        out = attention_out.transpose(0, 1)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# Example usage:
# images_tensor = torch.randn(1, 3, 224, 224)  # Example image tensor
# model = ViT_Res()
# output = model(images_tensor)
# print(output.shape)
