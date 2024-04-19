import torch
import torchvision
from tqdm import tqdm
import random
import os
from eval import *
from torch.utils.data import ConcatDataset
from baseline_model import BaselineModel

# Set seed
random.seed(12450)

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
    # Create weights folder if it doesn't exist yet
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
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
    model = BaselineModel()
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    train_data = torchvision.datasets.ImageFolder(train_path, transform=model.transforms)
    val_data = torchvision.datasets.ImageFolder(val_path, transform=model.transforms)
    # Loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_data, val_data, criterion, optimizer, num_epochs=num_epochs, save_path=save_path)
    # Save the model
    print("=== Saving model weights...===")
    torch.save(model.state_dict(), save_path)
    print("=== Model weights saved ===")
    return


def train_on_all_dataset(save_path, dataset_paths, criterion, num_epochs=5, model_path=None):
    print(f"=== Training on {save_path} ===")
    # Create weights folder if it doesn't exist yet
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    model = BaselineModel()
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize lists to hold all training and validation datasets
    all_train_data = []
    all_val_data = []

    # Load and transform data from all provided dataset paths
    transform = model.transforms
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


# Train a model on all GenImage datasets
path = ['resized_images/imagenet_ai_0419_biggan', 'resized_images/imagenet_ai_0419_sdv4', 'resized_images/imagenet_ai_0424_wukong', 'resized_images/imagenet_ai_0419_vqdm', \
        'resized_images/imagenet_glide', 'resized_images/imagenet_ai_0508_adm', 'resized_images/imagenet_ai_0424_sdv5', 'resized_images/imagenet_midjourney']
train_on_all_dataset('baseline_model/all_models.pth', path, criterion, num_epochs=1)

# Train a model on a each GenImage dataset separately
train_on_dataset('baseline_model/midjourney_model.pth', 'resized_images/imagenet_midjourney', criterion, num_epochs=3)
train_on_dataset('baseline_model/sdv5_model.pth', 'resized_images/imagenet_ai_0424_sdv5', criterion, num_epochs=3)
train_on_dataset('baseline_model/glide_model.pth', 'resized_images/imagenet_glide', criterion, num_epochs=3)
train_on_dataset('baseline_model/adm_model.pth', 'resized_images/imagenet_ai_0508_adm', criterion, num_epochs=3)
train_on_dataset('baseline_model/vqdm_model.pth', 'resized_images/imagenet_ai_0419_vqdm', criterion, num_epochs=3)
train_on_dataset('baseline_model/wukong_model.pth', 'resized_images/imagenet_ai_0424_wukong', criterion, num_epochs=3)
train_on_dataset('baseline_model/sdv4_model.pth', 'resized_images/imagenet_ai_0419_sdv4', criterion, num_epochs=3)
train_on_dataset('baseline_model/biggan_model.pth', 'resized_images/imagenet_ai_0419_biggan', criterion, num_epochs=3) 

# Evaluate on all models
# evaluate_all_baseline_model('resized_images', 'baseline_model')

# Evaluate on the Pixiv dataset (only the model trained on all GenImage datasets, in this case)
# model = BaselineModel()
# model.load_state_dict(torch.load('baseline_model/all_models.pth'))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# evaluate_pixiv(model, device, transform=BaselineModel().transforms)