import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime

class OCRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            # === FIRST BLOCK ====
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # === NEXT BLOCK ====
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # === Linear Flatten ===
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 26)
        )
    
    def forward(self, x):
        return self.stack(x)

if __name__ == "__main__":

    print(f"[{datetime.now()}][Info] Setting data transformation Formula... ")
    # AI model training retrieval
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print(f"[{datetime.now()}][Status] Downloading Dataset... Pulling from MNIST dataset..")
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    print(f"[{datetime.now()}][Status] Done, preparing data for training...")
    train_loader = DataLoader(dataset=train_dataset, batch_size = 64, shuffle=True, num_workers=0, pin_memory=True)

    # === Initialize Trainer ===
    print(f"[{datetime.now()}][Status] Data retrieved.  Initializing Model Trainer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCRModel()
    model = OCRModel().to(device)
    print(f"[{datetime.now()}][Status] Loaded Model")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"[{datetime.now()}][Status] Criterion and Optimizer Formulas have been loaded.  ")
    epochs = 10

    print(f"[{datetime.now()}][Status] Initializing Training Loop for {epochs} epochs. ")
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct/total
        print(f"[TRAINING] Epoch {epoch+1}/{epochs}, Loss is {epoch_loss:.4f}, Accuracy, {epoch_acc:.2f}%")

    print(f"[{datetime.now()}][Status] Training Done")
    print(f"[{datetime.now()}][Info] Saving Model to disk...")
    torch.save(model, "OCR_Model.pth")
    print(f"[{datetime.now()}][Status] Done! Exiting program...")
    exit(0)
