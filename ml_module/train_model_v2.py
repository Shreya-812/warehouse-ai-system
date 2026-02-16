import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------
# Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------
# Load CIFAR10
# ---------------------------
trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ---------------------------
# Label Remapping
# ---------------------------
# CIFAR original classes:
# 0 airplane
# 1 automobile
# 3 cat
# 5 bottle
# 9 truck

def remap_label(label):

    if label in [1, 9]:      # automobile, truck
        return 0             # heavy

    elif label in [5, 3]:    # bottle, cat
        return 1             # fragile

    elif label == 0:         # airplane
        return 2             # hazardous

    else:
        return None


class FilteredDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):

        self.dataset = dataset
        self.indices = []

        for i in range(len(dataset)):
            label = dataset.targets[i]
            if remap_label(label) is not None:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        img, label = self.dataset[self.indices[idx]]
        new_label = remap_label(label)

        return img, new_label


train_ds = FilteredDataset(trainset)
test_ds = FilteredDataset(testset)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=64, shuffle=False
)

print("Training samples:", len(train_ds))
print("Testing samples:", len(test_ds))


# ---------------------------
# Model
# ---------------------------
model = models.mobilenet_v2(weights="DEFAULT")

for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.last_channel, 3)

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

EPOCHS = 5


# ---------------------------
# Training
# ---------------------------
for epoch in range(EPOCHS):

    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {running_loss:.3f}")
    print(f"Accuracy: {acc:.2f}%")
    print("-"*30)


# ---------------------------
# Evaluation
# ---------------------------
model.eval()
y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())


labels_map = ["heavy", "fragile", "hazardous"]

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels_map))


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=labels_map,
            yticklabels=labels_map)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


torch.save(model.state_dict(), "warehouse_classifier.pth")
print("Model saved.")
