import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import glob
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

##################################################################################################


class DenoiseNet(nn.Module):
    def __init__(self):
        super(DenoiseNet, self).__init__()
        # Define a high-pass filter to enhance high-frequency components
        self.high_pass_filter = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)
        high_pass_kernel = torch.tensor([[[-1, -1, -1],
                                          [-1, 8, -1],
                                          [-1, -1, -1]]], dtype=torch.float32)
        # Expand dimensions to match (out_channels, in_channels/groups, height, width)
        self.high_pass_filter.weight.data = high_pass_kernel.repeat(3, 1, 1, 1)
        self.high_pass_filter.weight.requires_grad = False
        
        # Apply spectral normalization to convolutional layers
        self.conv1 = spectral_norm(nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1))
        self.conv2 = spectral_norm(nn.Conv2d(16, 32, kernel_size=3, padding=2, dilation=2, stride=1))
        self.conv3 = spectral_norm(nn.Conv2d(32, 64, kernel_size=3, padding=4, dilation=4, stride=1))
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply high-pass filtering to enhance high-frequency noise details
        x = self.high_pass_filter(x)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x


# LNP Model integrating DenoiseNet
class LNPModel(nn.Module):
    def __init__(self, denoise_net):
        super(LNPModel, self).__init__()
        self.denoise_net = denoise_net
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 32, 512),  # Adjust the size according to your output
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.denoise_net(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Dataset preparation
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.real_images = [os.path.join(root_dir, 'nature', x) for x in os.listdir(os.path.join(root_dir, 'nature'))]
        self.fake_images = [os.path.join(root_dir, 'ai', x) for x in os.listdir(os.path.join(root_dir, 'ai'))]
        self.all_images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_name = self.all_images[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

##################################################################################################


def find_all_val_dirs(root_dir):
    """Find all 'val' directories within the root directory."""
    return glob.glob(os.path.join(root_dir, '**/val'), recursive=True)


# Data Loading
def load_data(directory):
    dataset = ImageDataset(root_dir=directory, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    return loader


# Training Function
def train_model(loader, val_loader, model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training Phase
        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train

        # Validation Phase
        model.eval()  # Ensure the model is in evaluation mode
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Training Accuracy: {train_accuracy}%, Validation Accuracy: {val_accuracy}%')
        model.train()  # Switch back to training mode


# Evaluate the model on the validation data
def evaluate_dir(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    label_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            label_list.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    ruc = roc_auc_score(label_list, predictions)
    ap = average_precision_score(label_list, predictions)
    print(f"ROC AUC: {ruc:.2f}")
    print(f"Average Precision: {ap:.2f}")
    # print(f"Confusion Matrix: {confusion_matrix(label_list, predictions)}")
    return accuracy


def evaluate(root_dir, model, device):
    # Find all validation directories
    val_dirs = find_all_val_dirs(root_dir)
    total_accuracy = 0
    # Evaluate model on each validation directory
    for val_dir in val_dirs:
        print(f"Evaluating on {val_dir}")
        dataloader = load_data(val_dir)
        accuracy = evaluate_dir(model, dataloader, device)
        print(f"Accuracy for {val_dir}: {accuracy:.2f}%")
        total_accuracy += accuracy
    # report total accuracy
    total_accuracy = total_accuracy / len(val_dirs)
    print(f"Total accuracy: {total_accuracy:.2f}%")


def evaluateLNP_model(root_dir, model_path):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoise_net = DenoiseNet()
    model = LNPModel(denoise_net=denoise_net)
    model.load_state_dict(torch.load(model_path))
    denoise_net.to(device)
    model = model.to(device)
    evaluate(root_dir, model, device)


def evaluate_all_LNP_models(root_directory, models_directory):
    # List all files in the models directory
    model_files = [f for f in os.listdir(models_directory) if f.endswith('.pth')]

    # Loop through each model file and evaluate it
    for model_file in model_files:
        model_path = os.path.join(models_directory, model_file)
        print(f"Evaluating model: {model_path}")
        evaluateLNP_model(root_directory, model_path)


##################################################################################################


# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


##################################################################################################


if __name__ == '__main__':
    # # Datasets
    # datasets = ['ai_0419_biggan', 'ai_0419_sdv4', 'ai_0419_vqdm', 'ai_0424_sdv5',
    #             'ai_0424_wukong', 'ai_0508_adm', 'glide', 'midjourney']

    # for dataset in datasets:
    #     # Setup Model
    #     denoise_net = DenoiseNet().to(device)
    #     lnp_model = LNPModel(denoise_net=denoise_net).to(device)
    #     optimizer = optim.Adam(lnp_model.parameters(), lr=0.001)

    #     # Setup Datasets
    #     train_loader = load_data(f'resized_images/imagenet_{dataset}/train')
    #     val_loader = load_data(f'resized_images/imagenet_{dataset}/val')

    #     # Train The Model
    #     train_model(train_loader, val_loader, lnp_model, criterion, optimizer)

    #     # Save only the state dictionary
    #     print("=== Saving model weights...===")
    #     torch.save(lnp_model.state_dict(), f'lnp_model_{dataset}.pth')
    #     print("=== Model weights saved ===")

        # # Later to load the state dictionary into the model’s architecture
        # model = LNPModel(denoise_net=DenoiseNet())  # Recreate the model structure
        # model.load_state_dict(torch.load('lnp_model.pth'))
        # model.eval()  # Set the model to evaluation mode

    
    root_directory = 'resized_images'
    models_directory = './'
    evaluate_all_LNP_models(root_directory, models_directory)
