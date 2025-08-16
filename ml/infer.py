import io
import os
from typing import Optional
import torch
from PIL import Image
from torchvision import transforms
from .cnn import SimpleCNN


_loaded = None


def load_model_once(ckpt_path: Optional[str] = None):
    global _loaded
    if _loaded is not None:
        return _loaded
    if not ckpt_path:
        ckpt_path = os.getenv('CKPT_PATH')
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise RuntimeError('CKPT_PATH not set or file not found')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    _loaded = (model, device)
    return _loaded


def preprocess(img: Image.Image):
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return tfm(img).unsqueeze(0)


def infer_image(model_pack, file_obj) -> int:
    model, device = model_pack
    if hasattr(file_obj, 'read'):
        img = Image.open(file_obj).convert('RGB')
    else:
        img = Image.open(io.BytesIO(file_obj)).convert('RGB')
    x = preprocess(img).to(device)
    with torch.inference_mode():
        logits = model(x)
        pred = logits.argmax(dim=1).item()
    return int(pred)


