import io
from pathlib import Path

import torch
from PIL import Image
from fastapi import HTTPException
from torchvision import transforms

from api.batchnorm_dropout_cnn import BatchNormDropoutCNN

MODEL_PATH = Path(__file__).parent.parent.resolve() / "models" / "batchnorm_dropout_cnn.pt"

model = BatchNormDropoutCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

CLASSES = ["cat", "dog"]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="provided file is not an image")

    tensor = transform(image).unsqueeze(0) # es batchebis gamo damchirda (possible check)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    index = probs.argmax().item()
    return image, CLASSES[index], probs[index].item()
