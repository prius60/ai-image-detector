import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from ViT_model import ViTModel
from ViT_Res import ViT_Res
import os
import glob
from transformers import CLIPProcessor, CLIPModel

    
def find_all_val_dirs(root_dir):
    """Find all 'val' directories within the root directory."""
    return glob.glob(os.path.join(root_dir, '**/val'), recursive=True)

def load_data(val_dir, batch_size=32):
    """Load validation data from a given directory."""
    transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    dataloader = datasets.ImageFolder(val_dir, transform=transform)
    dataloader = DataLoader(dataloader, batch_size=batch_size, shuffle=False)
    return dataloader

def evaluate_dir(model, dataloader, device):
    """Evaluate the model on the validation data."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).squeeze(1).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
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

    transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    dataloader =  datasets.ImageFolder('pixiv_transformed', transform=transform)
    train_size = int(0.8 * len(dataloader))
    val_size = int(0.1 * len(dataloader))
    train_data, val_data, test_data = torch.utils.data.random_split(dataloader, [train_size, val_size, len(dataloader) - train_size - val_size])
    dataloader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)
    accuracy = evaluate_dir(model, dataloader, device)
    print(f"Accuracy for pixiv_transformed: {accuracy:.2f}%")

def evaluate_resnet(root_dir, model_path):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=False)  # Set pretrained=False as we'll load our own parameters
    model.fc = torch.nn.Linear(model.fc.in_features, 1 )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    evaluate(root_dir, model, device)
    
def evaluateViT(root_dir, model_path):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTModel()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    evaluate(root_dir, model, device)

def evaluateViT_Res(root_dir, model_path):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT_Res()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    evaluate(root_dir, model, device)


    
# # Example usage
# root_directory = 'resized_images'
# model_path = 'baseline_model.pth'
# evaluate_resnet(root_directory, model_path)

def evaluate_all_ViT(root_directory, models_directory):
    # List all files in the models directory
    model_files = [f for f in os.listdir(models_directory) if f.endswith('.pth')]
    
    # Loop through each model file and evaluate it
    for model_file in model_files:
        model_path = os.path.join(models_directory, model_file)
        print(f"Evaluating model: {model_path}")
        evaluateViT(root_directory, model_path)

def evaluate_all_ViT_Res(root_directory, models_directory):
    # List all files in the models directory
    model_files = [f for f in os.listdir(models_directory) if f.endswith('.pth')]
    
    # Loop through each model file and evaluate it
    for model_file in model_files:
        model_path = os.path.join(models_directory, model_file)
        print(f"Evaluating model: {model_path}")
        evaluateViT_Res(root_directory, model_path)

# # Example usage
# root_directory = 'resized_images'
# models_directory = './'
# evaluate_all_models(root_directory, models_directory)
