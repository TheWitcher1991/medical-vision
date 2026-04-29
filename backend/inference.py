from torch import Tensor
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_MAP = {
    "normal": "Норма",
    "tumor_glioma": "Глиома (опухоль)",
    "tumor_meningioma": "Менингиома (опухоль)",
    "tumor_pituitary": "Опухоль гипофиза",
}

CLASS_NAMES = ["normal", "tumor_glioma", "tumor_meningioma", "tumor_pituitary"]


def preprocess_image(image_bytes: bytes) -> Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    x = torch.tensor(list(image.getdata())).reshape(224, 224, 3)
    x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    x = (x - mean) / std

    return x

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict(image_bytes: bytes, model: torch.nn.Module) -> tuple:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        all_probs = probs.cpu().numpy()[0]
        
        pred_class = CLASS_NAMES[idx.item()]
        confidence = conf.item()
        probabilities = {CLASS_NAMES[i]: float(all_probs[i]) for i in range(len(CLASS_NAMES))}
    
    return pred_class, confidence, probabilities