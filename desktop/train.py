import json
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


CLASS_NAMES = ["normal", "tumor_glioma", "tumor_meningioma", "tumor_pituitary"]

NUM_CLASSES = len(CLASS_NAMES)


CONFIG = {
    "img_size": 224,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "epochs": EPOCHS,
    "backbone": "resnet18",
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
}


LABEL_MAP = {
    "yes": "tumor",
    "no": "normal",
    "glioma": "tumor_glioma",
    "meningioma": "tumor_meningioma",
    "pituitary": "tumor_pituitary",
    "notumor": "normal",
}


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


class MappedImageFolder(ImageFolder):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        original_class = self.classes[y]
        mapped_class = LABEL_MAP.get(original_class, None)

        if mapped_class is None:
            raise ValueError(f"Unknown class: {original_class}")

        new_label = CLASS_NAMES.index(mapped_class)

        return x, new_label


def load_dataset(path):

    train_path = os.path.join(path, "Training")
    test_path = os.path.join(path, "Testing")

    train_dataset = MappedImageFolder(train_path, transform=transform)
    test_dataset = MappedImageFolder(test_path, transform=transform)

    print("\n📊 Mapped classes:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"[{i}] {cls}")

    print("\n📈 Train distribution:")

    train_counts = Counter([y for _, y in train_dataset])
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls}: {train_counts[i]}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    return train_loader, test_loader


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


def train():

    model = MedClsNet(num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    datasets = [
        "masoudnickparvar/brain-tumor-mri-dataset",
        "volodymyrpivoshenko/skin-cancer-lesions-segmentation",
    ]

    for dataset_name in datasets:
        print(f"\n📦 Downloading: {dataset_name}")

        path = dataset_download(dataset_name)

        print(f"📂 Path: {path}")

        train_loader, test_loader = load_dataset(path)

        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, optimizer, criterion)

            print(f"[{dataset_name}] Epoch {epoch + 1}/{EPOCHS} | Loss: {loss:.4f}")

    torch.save(model.state_dict(), "medclsnet.pth")

    with open("medclsnet_config.json", "w") as f:
        json.dump(CONFIG, f, indent=4)

    print("\n✅ Model saved: medclsnet.pth")
    print("🧾 Config saved")


if __name__ == "__main__":
    train()
