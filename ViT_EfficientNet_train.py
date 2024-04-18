import torch
import torchvision
from tqdm import tqdm
import random
import os
from eval import *
from ViT_EfficientNet import ViT_EfficientNet


# Use CUDA for acceleration if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
transform = transforms.Compose([
    transforms.CenterCrop(224),     # Crop the center of the image
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP's specific normalization values
                         std=[0.26862954, 0.26130258, 0.27577711])
])

# Define loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()


# Define accuracy function
def accuracy(model, dataset, max_batches=None):
    correct, total = 0, 0
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).squeeze(1).float()
            correct += (predicted == labels.to(device)).sum().item()
            total += labels.size(0)
            if max_batches and total >= max_batches * dataloader.batch_size:
                break
    return correct / total


# Training loop
def train_model(model, train_data, val_data, criterion, optimizer, num_epochs=5, save_path=None):
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    for epoch in range(num_epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
        # Evaluation on the validation set
        val_acc = accuracy(model, val_data)
        print(f"Validation Accuracy after Epoch {epoch + 1}: {val_acc:.4f}")
        if save_path:
            torch.save(model.state_dict(), save_path)
    return model


def train_on_dataset(save_path, dataset_path, criterion, num_epochs=5, model_path=None):
    print(f"=== Training on {save_path} ===")
    # Create weights folder if it doesn't exist yet
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    model = ViT_EfficientNet()  # Assuming ViTModel is already defined.
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    train_data = torchvision.datasets.ImageFolder(train_path, transform=transform)
    val_data = torchvision.datasets.ImageFolder(val_path, transform=transform)
    # Loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_data, val_data, criterion, optimizer, num_epochs=num_epochs, save_path=save_path)
    # Save the model
    print("=== Saving model weights...===")
    torch.save(model.state_dict(), save_path)
    print("=== Model weights saved ===")
    return


# train_on_dataset('ViT_EfficientNet_weights/midjourney_model.pth', 'resized_images/imagenet_midjourney', criterion, num_epochs=3)
# train_on_dataset('ViT_EfficientNet_weights/sdv5_model.pth', 'resized_images/imagenet_ai_0424_sdv5', criterion, num_epochs=3)
# train_on_dataset('ViT_EfficientNet_weights/glide_model.pth', 'resized_images/imagenet_glide', criterion, num_epochs=3)
# train_on_dataset('ViT_EfficientNet_weights/adm_model.pth', 'resized_images/imagenet_ai_0508_adm', criterion, num_epochs=3)
# train_on_dataset('ViT_EfficientNet_weights/vqdm_model.pth', 'resized_images/imagenet_ai_0419_vqdm', criterion, num_epochs=3)
# train_on_dataset('ViT_EfficientNet_weights/wukong_model.pth', 'resized_images/imagenet_ai_0424_wukong', criterion, num_epochs=3)
# train_on_dataset('ViT_EfficientNet_weights/sdv4_model.pth', 'resized_images/imagenet_ai_0419_sdv4', criterion, num_epochs=3)
train_on_dataset('ViT_EfficientNet_weights/biggan_model.pth', 'resized_images/imagenet_ai_0419_biggan', criterion, num_epochs=3) 