import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
# ── 1. Transforms ────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# ── 2. Dataset ───────────────────────────────────────────
train_dataset = datasets.CIFAR10(root='./data', train=True,  transform=train_transform, download=True)
test_dataset  = datasets.CIFAR10(root='./data', train=False, transform=transform,       download=True)

# ── 3. DataLoader ────────────────────────────────────────
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  pin_memory=True, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, pin_memory=True, num_workers=2)

# ── 4. Model ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        out = out + x          # skip connection
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 → 32  |  32x32 → 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResBlock(32),
            nn.MaxPool2d(2),

            # Block 2: 32 → 64  |  16x16 → 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
            nn.MaxPool2d(2),

            # Block 3: 64 → 128  |  8x8 → 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            nn.MaxPool2d(2),

            # Block 4: 128 → 256  |  4x4 → 2x2
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResBlock(256),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),           # 256*2*2 = 1024
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ── 5. Setup ─────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = ResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ── 6. Training loop ─────────────────────────────────────
for epoch in range(30):
    model.train()
    total_loss, correct = 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * correct / ((loop.n+1) * 128):.2f}%"
        )

    print(f"Epoch {epoch+1:02d} done | Acc: {100 * correct / len(train_dataset):.2f}%")

# ── 7. Evaluation ────────────────────────────────────────
model.eval()
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()

print(f"Test Accuracy: {100 * correct / len(test_dataset):.2f}%")