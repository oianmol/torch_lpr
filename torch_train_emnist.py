import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class EMNISTCNN(nn.Module):
    def __init__(self, num_classes):
        super(EMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EMNISTTrainer:
    def __init__(self, batch_size=36, num_epochs=10, device=None):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.transform = self.build_transform()
        self._load_data()
        self.model = EMNISTCNN(self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def build_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: TF.rotate(x, -90)),
            transforms.Lambda(lambda x: TF.hflip(x)),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _load_data(self):
        try:
            self.train_dataset = self.training()
            self.test_dataset = self.testing()
        except RuntimeError as e:
            print(f"Error loading EMNIST: {e}")
            exit()
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.num_classes = self.batch_size
        self.emnist_char_labels = self.train_dataset.classes[:self.batch_size]

    def testing(self):
        return torchvision.datasets.EMNIST(
            root='./emnist_data',
            split='balanced',
            train=False,
            download=True,
            transform=self.transform
        )

    def training(self):
        return torchvision.datasets.EMNIST(
            root='./emnist_data',
            split='balanced',
            train=True,
            download=True,
            transform=self.transform
        )

    def train(self):
        print(f"Training on: {self.device}")
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = correct_predictions / total_samples
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Test Accuracy: {100 * correct / total:.2f}%')

    def save(self, path='emnist_torchvision.pth'):
        torch.save(self.model.state_dict(), path)

    def load_and_predict(self, saved_model_path='emnist_torchvision.pth', batch_size=36, device=None):
        device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # Use the same transforms as during training
        transform = self.build_transform()
        test_dataset = torchvision.datasets.EMNIST(
            root='./emnist_data',
            split='balanced',
            train=False,
            download=True,
            transform=transform
        )
        emnist_char_labels = test_dataset.classes
        model = EMNISTCNN(num_classes=batch_size).to(device)
        model.load_state_dict(torch.load(saved_model_path, map_location=device))
        model.eval()
        # Pick a random test sample
        import numpy as np
        idx = np.random.randint(batch_size)
        sample_image_tensor, true_label_idx = test_dataset[idx]
        sample_image_tensor = sample_image_tensor.to(device)
        with torch.no_grad():
            output = model(sample_image_tensor.unsqueeze(0))
            _, predicted_label_idx = torch.max(output.data, 1)
        predicted_char = emnist_char_labels[predicted_label_idx.item()]
        true_char = emnist_char_labels[true_label_idx]
        print(f"True Character: {true_char}")
        print(f"Predicted Character: {predicted_char}")
        plt.imshow(sample_image_tensor.cpu().squeeze().numpy() / 2 + 0.5, cmap='gray')
        plt.title(f"True: {true_char}, Predicted: {predicted_char}")
        plt.axis('off')
        plt.show()

    def predict_and_show(self):
        idx = np.random.randint(len(self.test_dataset))
        sample_image_tensor, true_label_idx = self.test_dataset[idx]
        sample_image_tensor = sample_image_tensor.to(self.device)
        with torch.no_grad():
            self.model.eval()
            output = self.model(sample_image_tensor.unsqueeze(0))
            _, predicted_label_idx = torch.max(output.data, 1)
        predicted_char = self.emnist_char_labels[predicted_label_idx.item()]
        true_char = self.emnist_char_labels[true_label_idx]
        print(f"\nTrue Character: {true_char}")
        print(f"Predicted Character: {predicted_char}")
        plt.imshow(sample_image_tensor.cpu().squeeze().numpy() / 2 + 0.5, cmap='gray')
        plt.title(f"True: {true_char}, Predicted: {predicted_char}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    trainer = EMNISTTrainer(batch_size=36)
    trainer.train()
    trainer.evaluate()
    trainer.save()
    trainer.predict_and_show()