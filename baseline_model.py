import torch
import torchvision
import random
from tqdm import tqdm

# Set seed
random.seed(12450)


# Load pre-trained model
model = torchvision.models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# Prepare transformation of images
transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()

# Use CUDA for acceleration if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load dataset and split into train and val sets
dataset = torchvision.datasets.ImageFolder('F:\\pixiv_transformed', transform=transform)
# dataset = torch.utils.data.Subset(dataset, random.sample(range(len(dataset)), 1000))
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, len(dataset) - train_size - val_size])

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Calculate accuracy
def accuracy(model, dataset, max_batches=None):
    correct, total = 0, 0
    iter_count = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        iter_count += 1

        if max_batches is not None and iter_count >= max_batches:
            break
    
    return correct / total
    

# Training loop
def train_model(model, train_data, val_data, criterion, optimizer, num_epochs=5, validate_every=100):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    iter_count = 0
    print("===== Training Started =====")

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Print statistics
            iter_count += 1
            if iter_count % validate_every == 0:
                train_accuracy = accuracy(model, train_data, max_batches=25)
                val_accuracy = accuracy(model, val_data, max_batches=25)
                print(f'Iter {iter_count}, Loss: {loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
                running_loss = 0.0

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_data)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Acc: {accuracy(model, val_data):.4f}')

    print("===== Finished Training =====")
    return model

# Train the model
model = train_model(model, train_data, val_data, criterion, optimizer, num_epochs=5, validate_every=200)

# Test the model
test_accuracy = accuracy(model, test_data)
print(f'Final Test Accuracy: {test_accuracy:.4f}')

# Save the model
print("=== Saving model weights...===")
torch.save(model.state_dict(), 'baseline_model.pth')
print("=== Model weights saved ===")

