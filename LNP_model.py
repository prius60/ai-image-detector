import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

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


# Data Loading
def load_data(directory):
    dataset = ImageDataset(root_dir=directory, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    return loader

# Training Function
def train_model(loader, model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('training')
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(loader)}')


##################################################################################################


# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denoise_net = DenoiseNet().to(device)
lnp_model = LNPModel(denoise_net=denoise_net).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lnp_model.parameters(), lr=0.001)


##################################################################################################


if __name__ == '__main__':
    train_loader = load_data('imagenet_ai_0419_biggan/train')
    print('made it here')
    train_model(train_loader, lnp_model, criterion, optimizer)

    # Save only the state dictionary
    print("=== Saving model weights...===")
    torch.save(lnp_model.state_dict(), 'lnp_model.pth')
    print("=== Model weights saved ===")

    # # Later to load the state dictionary into the model’s architecture
    # model = LNPModel(denoise_net=DenoiseNet())  # Recreate the model structure
    # model.load_state_dict(torch.load('lnp_model.pth'))
    # model.eval()  # Set the model to evaluation mode














# class DenoiseNet(nn.Module):
#     def __init__(self):
#         super(DenoiseNet, self).__init__()
#         # Define a high-pass filter to enhance high-frequency components
#         self.high_pass_filter = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
#         self.high_pass_filter.weight.data = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]] * 3).float()
#         # Ensure the high-pass filter does not get updated during training
#         self.high_pass_filter.weight.requires_grad = False
        
#         # Apply spectral normalization to convolutional layers to stabilize their training
#         self.conv1 = spectral_norm(nn.Conv2d(3, 16, kernel_size=3, padding=1))
#         self.conv2 = spectral_norm(nn.Conv2d(16, 32, kernel_size=3, padding=2, dilation=2))  # Dilated convolution
#         self.conv3 = spectral_norm(nn.Conv2d(32, 64, kernel_size=3, padding=4, dilation=4))  # Further dilated convolution
#         self.pool = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         # Apply high-pass filtering to enhance high-frequency noise details
#         x = self.high_pass_filter(x)
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = self.pool(self.relu(self.conv3(x)))
#         return x




# # Define DenoiseNet (simplified version for illustration)
# class DenoiseNet(nn.Module):
#     def __init__(self):
#         super(DenoiseNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = self.pool(self.relu(self.conv3(x)))
#         return x















# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torchvision import datasets, transforms
# from PIL import Image

# def high_pass_filter(image):
#     dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
#     dft_shift = np.fft.fftshift(dft)
#     rows, cols = image.shape[:2]
#     crow, ccol = rows//2, cols//2
#     # Create a mask with a small central square set to zero (high-pass filter)
#     mask = np.ones((rows, cols, 2), np.uint8)
#     mask[crow-30:crow+30, ccol-30:ccol+30] = 0
#     fshift = dft_shift * mask
#     f_ishift = np.fft.ifftshift(fshift)
#     img_back = cv2.idft(f_ishift)
#     img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
#     return img_back


# class LNPDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.dataset = datasets.ImageFolder(root=root, transform=transform)
    
#     def __getitem__(self, index):
#         img, label = self.dataset[index]
#         img = high_pass_filter(np.array(img))  # Apply high-pass filter
#         img = Image.fromarray(np.uint8(img))   # Convert back to PIL image for further transformations
#         if self.transform:
#             img = self.transform(img)
#         return img, label

#     def __len__(self):
#         return len(self.dataset)

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# import torch.nn as nn

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 16, 3, padding=1),  # Input is 1 channel image
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(64 * 32 * 32, 128),
#             nn.ReLU(),
#             nn.Linear(128, 2),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

# def train_model(model, train_loader, val_loader, epochs, device):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     model.train()
#     for epoch in range(epochs):
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
        
#         # Validation accuracy
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         print(f'Epoch {epoch+1}, Validation Accuracy: {100 * correct / total}%')


# def evaluate_model(model, test_loader, device):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = 100 * correct / total
#     print(f'Test Accuracy: {accuracy}%')
#     return accuracy




# # Step 1
# import torch
# import cv2
# import numpy as np
# import torch.nn as nn
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# import os
# from PIL import Image

# def high_pass_filter(image):
#     # Apply DFT to each channel independently
#     channels = cv2.split(image)
#     result_channels = []
#     for channel in channels:
#         dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
#         dft_shift = np.fft.fftshift(dft)

#         rows, cols = channel.shape
#         crow, ccol = rows // 2, cols // 2

#         # Create a mask first, center square is 1, remaining all zeros
#         mask = np.zeros((rows, cols, 2), np.uint8)
#         r = 30  # Radius of the low frequencies to block
#         center = [crow, ccol]
#         x, y = np.ogrid[:rows, :cols]
#         mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
#         mask[mask_area] = 0

#         # Apply mask and inverse DFT
#         fshift = dft_shift * mask
#         f_ishift = np.fft.ifftshift(fshift)
#         img_back = cv2.idft(f_ishift)
#         img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

#         # Normalize to 0-255 and convert to uint8
#         cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
#         result_channels.append(img_back)

#     # Merge channels back to a color image
#     processed_image = cv2.merge(result_channels)
#     processed_image = np.uint8(processed_image)

#     return processed_image





# # Step 2
# class LNPDataset(Dataset):
#     def __init__(self, root, transform=None):
#         super().__init__()
#         self.root = root
#         self.transform = transform
#         self.classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]  # Filter to include only directories
#         self.files = [os.path.join(root, cls, file) for cls in self.classes for file in os.listdir(os.path.join(root, cls)) if not file.startswith('.')]  # Ignore hidden files

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         file_path = self.files[idx]
#         label = self.classes.index(os.path.basename(os.path.dirname(file_path)))
#         img = Image.open(file_path).convert('RGB')
#         img = np.array(img)
#         img = high_pass_filter(img)  # Apply your high-pass filter here
#         img = Image.fromarray(np.uint8(img))  # Convert back to PIL image for further transformations
#         if self.transform:
#             img = self.transform(img)
#         return img, label
    

# # Step 3
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 16, 3, padding=1),  # Input is 1 channel image
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(64 * 32 * 32, 128),
#             nn.ReLU(),
#             nn.Linear(128, 2),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
    

# # Step 4
# def train_model(model, train_loader, val_loader, epochs, device):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     model.train()
#     for epoch in range(epochs):
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
        
#         # Validation accuracy
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         print(f'Epoch {epoch+1}, Validation Accuracy: {100 * correct / total}%')


# # Step 5
# def evaluate_model(model, test_loader, device):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = 100 * correct / total
#     print(f'Test Accuracy: {accuracy}%')
#     return accuracy



# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Set the paths
# train_dir = '/Users/raine/Desktop/datasets/resized_images/imagenet_ai_0419_biggan/train'
# val_dir = '/Users/raine/Desktop/datasets/resized_images/imagenet_ai_0419_biggan/val'

# train_dataset = LNPDataset(train_dir, transform=transform)
# val_dataset = LNPDataset(val_dir, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SimpleCNN().to(device)

# # Define the training function here
# # Assume train_model function is defined as described earlier

# train_model(model, train_loader, val_loader, epochs=10, device=device)

