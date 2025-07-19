import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF # Import functional for direct use

# 1. Define Transformations
# EMNIST images are 28x28 grayscale.
# We need to convert them to PyTorch tensors and normalize.
# The `ToTensor()` transform will put pixel values in [0, 1].
# Note: torchvision's EMNIST generally loads correctly oriented,
# unlike some raw EMNIST files, but verify by plotting.
transform = transforms.Compose([
    transforms.ToTensor(),
    # --- Corrected Transforms for EMNIST Orientation ---
    # Apply rotation and horizontal flip directly using functional transforms within a Lambda
    # This ensures the order of operations and handles the specific EMNIST orientation.
    transforms.Lambda(lambda x: TF.rotate(x, -90)),  # Rotate 90 degrees clockwise
    transforms.Lambda(lambda x: TF.hflip(x)),      # Then flip horizontally
    # --------------------------------------------------
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] range for CNNs
])
device = torch.device("mps")

# 2. Load the EMNIST Dataset
# Specify a 'root' directory where the dataset will be stored.
# Set download=True to attempt download if not present.
# Choose your split ('byclass', 'byclass', 'letters', 'digits', 'mnist')
# For license plates, 'balanced' is a good starting point.
try:
    train_dataset = torchvision.datasets.EMNIST(
        root='./emnist_data',
        split='balanced',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.EMNIST(
        root='./emnist_data',
        split='balanced',
        train=False,
        download=True,
        transform=transform
    )

except RuntimeError as e:
    print(f"Error loading EMNIST with torchvision: {e}")
    print("This often happens if the download link is broken or the file is corrupted.")
    print("Try deleting the './emnist_data' folder and re-running.")
    print("If it persists, consider manual download or using ActiveLoop/TFDS.")
    exit() # Exit if dataset loading fails

# Create DataLoaders for batching and shuffling
BATCH_SIZE = 36
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Get number of classes
num_classes = BATCH_SIZE
print(f"Number of classes for EMNIST Balanced: {num_classes}")

# Get the EMNIST Balanced character mapping from the dataset's `classes` attribute
# Note: EMNIST in torchvision maps numerical labels to characters differently
# than the raw NIST mapping or TFDS's names list.
# `train_dataset.classes` will give you a list like ['0', '1', ..., 'A', 'B', ...]
emnist_char_labels = train_dataset.classes[:BATCH_SIZE]
print(f"EMNIST character mapping : {emnist_char_labels}")

# 3. Define Your PyTorch CNN Model
class EMNISTCNN(nn.Module):
    def __init__(self, num_classes):
        super(EMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # After two 2x2 pooling layers, 28x28 becomes 7x7
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = EMNISTCNN(num_classes)

# 4. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Train the Model
num_epochs = 10
model.to(device)

print(f"Training on: {device}")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# 6. Evaluate the Model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 7. Save the Model
torch.save(model.state_dict(), 'emnist_torchvision.pth')

# --- Example Prediction (Optional) ---
# Get a single image from the test set
sample_image_tensor, true_label_idx = test_dataset[np.random.randint(len(test_dataset))]
sample_image_tensor = sample_image_tensor.to(device)

# Make prediction
with torch.no_grad():
    model.eval()
    output = model(sample_image_tensor.unsqueeze(0)) # Add batch dimension
    _, predicted_label_idx = torch.max(output.data, 1)

predicted_char = emnist_char_labels[predicted_label_idx.item()]
true_char = emnist_char_labels[true_label_idx]

print(f"\nTrue Character: {true_char}")
print(f"Predicted Character: {predicted_char}")

# Display image
plt.imshow(sample_image_tensor.cpu().squeeze().numpy() / 2 + 0.5, cmap='gray') # Denormalize for display
plt.title(f"True: {true_char}, Predicted: {predicted_char}")
plt.axis('off')
plt.show()