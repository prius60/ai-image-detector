import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import os
from eval import *
from torch.utils.data import ConcatDataset
from ViT_Res_patch import ViT_Res_patch
from transformers import CLIPProcessor, CLIPModel
# Set seed for reproducibility
# random.seed(12450)


# Use CUDA for acceleration if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



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
        count = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            count += 1
            # if count % 100 == 0:
            #     val_acc = accuracy(model, val_data, max_batches=5)
            #     print(f"Validation Accuracy after {count} iterations: {val_acc:.4f}")
        # Evaluation on the validation set
        val_acc = accuracy(model, val_data, max_batches=25)
        print(f"Validation Accuracy after Epoch {epoch + 1}: {val_acc:.4f}")
        if save_path:
            torch.save(model.state_dict(), save_path)
    return model

def train_on_dataset(save_path, dataset_path, criterion, num_epochs=5, model_path=None):
    print(f"=== Training on {save_path} ===")
    model = ViT_Res_patch()  # Assuming ViTModel is already defined.
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    transform = transforms.Compose([
    transforms.CenterCrop(224),     # Crop the center of the image
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP's specific normalization values
                         std=[0.26862954, 0.26130258, 0.27577711])
    ])
    train_data = torchvision.datasets.ImageFolder(train_path, transform=transform)
    val_data = torchvision.datasets.ImageFolder(val_path, transform=transform)
    # Loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_data, val_data, criterion, optimizer, num_epochs=num_epochs, save_path=save_path)
    # Save the model
    print("=== Saving model weights...===")
    torch.save(model.state_dict(), save_path)
    print("=== Model weights saved ===")
    return
def train_on_pixiv(save_path, criterion, num_epochs=5, model_path=None):
    print(f"=== Training on {save_path} ===")
    model = ViT_Res_patch()  # Assuming ViTModel is already defined.
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    transform = transforms.Compose([
    transforms.CenterCrop(224),     # Crop the center of the image
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP's specific normalization values
                         std=[0.26862954, 0.26130258, 0.27577711])
    ])
    dataloader =  datasets.ImageFolder('pixiv_transformed', transform=transform)
    train_size = int(0.8 * len(dataloader))
    val_size = int(0.1 * len(dataloader))
    train_data, val_data, test_data = torch.utils.data.random_split(dataloader, [train_size, val_size, len(dataloader) - train_size - val_size])
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_data, val_data, criterion, optimizer, num_epochs=num_epochs, save_path=save_path)
    # Save the model
    print("=== Saving model weights...===")
    torch.save(model.state_dict(), save_path)
    print("=== Model weights saved ===")
    return
def train_on_all_dataset(save_path, dataset_paths, criterion, num_epochs=5, model_path=None):
    print(f"=== Training on {save_path} ===")
    model = ViT_Res_patch()  # Assuming ViTModel is already defined.
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize lists to hold all training and validation datasets
    all_train_data = []
    all_val_data = []

    # Load and transform data from all provided dataset paths
    transform = transforms.Compose([
    transforms.CenterCrop(224),     # Crop the center of the image
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP's specific normalization values
                         std=[0.26862954, 0.26130258, 0.27577711])
    ])
    for path in dataset_paths:
        train_path = os.path.join(path, 'train')
        val_path = os.path.join(path, 'val')
        train_data = torchvision.datasets.ImageFolder(train_path, transform=transform)
        val_data = torchvision.datasets.ImageFolder(val_path, transform=transform)
        all_train_data.append(train_data)
        all_val_data.append(val_data)

    # Combine all datasets
    combined_train_data = ConcatDataset(all_train_data)
    combined_val_data = ConcatDataset(all_val_data)

    # Loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, combined_train_data, combined_val_data, criterion, optimizer, num_epochs=num_epochs, save_path=save_path)
    
    # Save the model
    print("=== Saving model weights...===")
    torch.save(model.state_dict(), save_path)
    print("=== Model weights saved ===")
    return

# import time
# print("Program will start after 1 hour.")
# time.sleep(3600)  # Sleep for 3600 seconds (1 hour)

train_on_dataset('16patch/biggan_model.pth', 'resized_images/imagenet_ai_0419_biggan', criterion, num_epochs=3) 
train_on_dataset('16patch/sdv4_model.pth', 'resized_images/imagenet_ai_0419_sdv4', criterion, num_epochs=3)
train_on_dataset('16patch/wukong_model.pth', 'resized_images/imagenet_ai_0424_wukong', criterion, num_epochs=3)
train_on_dataset('16patch/vqdm_model.pth', 'resized_images/imagenet_ai_0419_vqdm', criterion, num_epochs=3)
train_on_dataset('16patch/glide_model.pth', 'resized_images/imagenet_glide', criterion, num_epochs=3)
train_on_dataset('16patch/adm_model.pth', 'resized_images/imagenet_ai_0508_adm', criterion, num_epochs=3)
train_on_dataset('16patch/sdv5_model.pth', 'resized_images/imagenet_ai_0424_sdv5', criterion, num_epochs=3)
train_on_dataset('16patch/midjourney_model.pth', 'resized_images/imagenet_midjourney', criterion, num_epochs=3)
train_on_pixiv('16patch/pixiv.pth', criterion, num_epochs=3)

# import time
# print("Program will start after 1 hour.")
# time.sleep(3600)  # Sleep for 3600 seconds (1 hour)

# # After 1 hour, execute the rest of the program
# print("Program has started.")


criterion = torch.nn.BCEWithLogitsLoss()
path = ['resized_images/imagenet_ai_0419_biggan', 'resized_images/imagenet_ai_0419_sdv4', 'resized_images/imagenet_ai_0424_wukong', 'resized_images/imagenet_ai_0419_vqdm', \
        'resized_images/imagenet_glide', 'resized_images/imagenet_ai_0508_adm', 'resized_images/imagenet_ai_0424_sdv5', 'resized_images/imagenet_midjourney']
train_on_all_dataset('16patch/all_models.pth', path, criterion, num_epochs=3)
# train_on_pixiv('./ViT_Res_pixiv/pixiv.pth', criterion, num_epochs=2)

# # Example usage
root_directory = 'resized_images'
models_directory = './16patch'

evaluate_all_ViT_Res_patch(root_directory, models_directory)