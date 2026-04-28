import os
from collections import Counter

import torch
import torch.nn as nn
from kagglehub import dataset_download
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import MedClsNet

os.environ["KAGGLE_API_TOKEN"] = "KGAT_b465f7cb751f72901abbb4220b551af0"
os.environ["KAGGLEHUB_CACHE"] = "./datasets"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 8

CONFIG = {
    "img_size": 224,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "epochs": EPOCHS,
    "backbone": "resnet18",
}

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def load_dataset(path):
    dataset = ImageFolder(root=path, transform=transform)

    class_names = dataset.classes

    print("\n📊 Dataset classes:")
    for i, cls in enumerate(class_names):
        print(f"  [{i}] {cls}")

    targets = [y for _, y in dataset.samples]

    counts = Counter(targets)

    print("\n📈 Class distribution:")

    for i, cls in enumerate(class_names):
        print(f"  {cls}: {counts[i]} images")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    return loader, class_names


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0

    loop = tqdm(loader, desc="Training", leave=False)

    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def train_on_datasets(dataset_list):
    model = MedClsNet(num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    all_classes = set()

    for dataset_name in dataset_list:
        print(f"\n📦 Downloading dataset: {dataset_name}")

        path = dataset_download(dataset_name)

        print(f"📂 Loaded at: {path}")

        loader, class_names = load_dataset(path)

        all_classes.update(class_names)

        print(f"🔢 Classes found: {len(class_names)}")

        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, loader, optimizer, criterion)

            print(f"[{dataset_name}] Epoch {epoch + 1}/{EPOCHS} - Loss: {loss:.4f}")

    # ----------------------------
    # SAVE MODEL
    # ----------------------------
    torch.save(model.state_dict(), "medclsnet_final.pth")

    # ----------------------------
    # SAVE CONFIG JSON
    # ----------------------------
    config = {
        **CONFIG,
        "num_classes": NUM_CLASSES,
        "class_names": sorted(list(all_classes)),
    }

    with open("medclsnet_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("✅ Model saved!")
    print("🧾 Config saved to medclsnet_config.json")


if __name__ == "__main__":
    datasets = [
        "navoneel/brain-mri-images-for-brain-tumor-detection",
        "masoudnickparvar/brain-tumor-mri-dataset",
    ]

    train_on_datasets(datasets)
