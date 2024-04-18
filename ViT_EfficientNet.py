import torch
import torchvision
from eval import *
import torch.nn as nn

from transformers import CLIPModel


class ViT_EfficientNet(nn.Module):
    def __init__(self, n_layers=3, n_heads=4):
        super(ViT_EfficientNet, self).__init__()
        self.ViT = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.efficientnet = torchvision.models.efficientnet_b4(weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT)
        
        # Project both outputs to the same dimension, here chosen as 128 for example
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(self.efficientnet.classifier[1].in_features, 128, bias=True),
        )
        self.ViT_fc = nn.Linear(768, 128)

        # Freeze CLIP ViT model parameters
        # Uncomment if you want to freeze ViT parameters
        for param in self.ViT.parameters():
            param.requires_grad = False

        # Freeze EfficientNet parameters
        # Uncomment if you want to freeze EfficientNet parameters
        for param in self.efficientnet.features.parameters():
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

        # Process input through EfficientNet
        efficientnet_outputs = self.efficientnet(images_tensor)

        # Concatenate the outputs along the feature dimension
        combined_features = torch.cat((vit_outputs, efficientnet_outputs), dim=1)
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
# model = ViT_EfficientNet()
# output = model(images_tensor)
# print(output.shape)
