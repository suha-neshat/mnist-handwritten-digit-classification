import io
import base64
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

# ---------------------------------------------------------------------
# Robust path handling: find mnist_cnn.py and mnist_cnn.pt regardless
# of where `uvicorn` is launched from (root/ or api/).
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../api
ROOT_DIR = BASE_DIR.parent                           # project root

# Ensure we can import MNISTCNN from either api/ or model/
if (BASE_DIR / "mnist_cnn.py").exists():
    sys.path.insert(0, str(BASE_DIR))
elif (ROOT_DIR / "model" / "mnist_cnn.py").exists():
    sys.path.insert(0, str(ROOT_DIR / "model"))

from mnist_cnn import MNISTCNN  # noqa: E402

# Candidate locations for the trained weights
MODEL_CANDIDATES = [
    BASE_DIR / "mnist_cnn.pt",             # api/mnist_cnn.pt
    ROOT_DIR / "model" / "mnist_cnn.pt",   # model/mnist_cnn.pt
]

model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
if model_path is None:
    raise FileNotFoundError(
        "Could not find 'mnist_cnn.pt'. Train first (python model/train.py). "
        f"Looked in: {', '.join(str(p) for p in MODEL_CANDIDATES)}"
    )

# ---------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Same normalization used during training
tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# ---------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------
def _b64_to_pil(data_url: str) -> Image.Image:
    """
    Accepts a data URL ("data:image/png;base64,...") or raw base64.
    Returns a PIL grayscale image.
    """
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")


def predict_from_b64(data_url: str):
    """
    Convert base64 image -> 28x28 tensor -> logits -> (digit, probabilities[])
    The canvas UI is white-on-black; MNIST is black-on-white, so we invert.
    """
    img = _b64_to_pil(data_url)
    img = ImageOps.invert(img)  # white-on-black -> black-on-white
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    return pred, [float(p) for p in probs.cpu().tolist()]
