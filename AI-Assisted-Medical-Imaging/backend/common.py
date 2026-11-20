
"""Shared model, preprocessing, DICOM reading and Grad-CAM code."""

import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models
from monai.transforms import Compose, Resize, ScaleIntensity


CLASSES = ["NO_LUNG_OPACITY", "PNEUMONIA"]

IMG_SIZE = 224

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# MONAI preprocessing
_base_tf = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ScaleIntensity(minv=0.0, maxv=1.0)
])


def to_tensor(x) -> torch.Tensor:
    """Convert MONAI MetaTensor or array to plain float tensor."""

    if hasattr(x, "as_tensor"):
        x = x.as_tensor()

    return torch.as_tensor(x).float()


# ------------------------------------------------------------
# Image preprocessing - memory optimized
# ------------------------------------------------------------

def load_unnormalised(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL image to 3x224x224 tensor.

    Resize the image before NumPy float32 conversion
    to reduce memory consumption.
    """

    # Convert to RGB
    img = img.convert("RGB")

    # Resize before allocating large NumPy arrays
    img = img.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR
    )

    # Convert resized image to NumPy
    arr = np.asarray(
        img,
        dtype=np.float32
    ).transpose(2, 0, 1)

    # MONAI intensity scaling
    return to_tensor(_base_tf(arr))


def normalise(x: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization."""

    return (x - MEAN) / STD


def pil_to_input(img: Image.Image) -> torch.Tensor:
    """Complete image preprocessing pipeline."""

    return normalise(load_unnormalised(img))


# ------------------------------------------------------------
# Device selection
# ------------------------------------------------------------

def get_device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ------------------------------------------------------------
# DenseNet121 model
# ------------------------------------------------------------

class DenseNetCAM(nn.Module):
    """
    DenseNet121 classifier with feature maps
    available for Grad-CAM.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True
    ):
        super().__init__()

        net = None

        if pretrained:
            try:
                net = models.densenet121(
                    weights=models.DenseNet121_Weights.DEFAULT
                )

            except Exception as e:
                print(
                    f"[warn] Could not load ImageNet weights: {e}"
                )

        if net is None:
            net = models.densenet121(weights=None)

        self.features = net.features

        self.classifier = nn.Linear(
            net.classifier.in_features,
            num_classes
        )

    def forward(self, x, return_maps: bool = False):

        act = F.relu(self.features(x))

        logits = self.classifier(
            F.adaptive_avg_pool2d(act, 1).flatten(1)
        )

        if return_maps:
            return logits, act

        return logits


# ------------------------------------------------------------
# Grad-CAM
# ------------------------------------------------------------

def grad_cam(model: DenseNetCAM, x: torch.Tensor, class_idx=None):
    """
    Generate Grad-CAM heatmap.

    Returns:
        cam: 224x224 heatmap
        probs: class probabilities
        class index: predicted or selected class
    """

    model.eval()

    with torch.enable_grad():

        logits, act = model(x, return_maps=True)

        probs = (
            F.softmax(logits, dim=1)[0]
            .detach()
            .cpu()
            .numpy()
        )

        c = (
            int(logits.argmax(1).item())
            if class_idx is None
            else class_idx
        )

        grads = torch.autograd.grad(
            logits[0, c],
            act
        )[0]

    weights = grads.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = F.relu(
        (weights * act).sum(1, keepdim=True)
    ).detach()

    cam = F.interpolate(
        cam,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    cam = cam - cam.min()

    cam = cam / (cam.max() + 1e-8)

    return cam.cpu().numpy(), probs, c


# ------------------------------------------------------------
# Heatmap generation
# ------------------------------------------------------------

def _jet(v: np.ndarray) -> np.ndarray:

    r = np.clip(
        1.5 - np.abs(4 * v - 3),
        0,
        1
    )

    g = np.clip(
        1.5 - np.abs(4 * v - 2),
        0,
        1
    )

    b = np.clip(
        1.5 - np.abs(4 * v - 1),
        0,
        1
    )

    return np.stack([r, g, b], axis=-1)


def heatmap_rgba(
    cam: np.ndarray,
    size: int = 448
) -> Image.Image:
    """
    Generate transparent Grad-CAM PNG.
    """

    c = Image.fromarray(
        (cam * 255).astype(np.uint8)
    ).resize(
        (size, size),
        Image.BILINEAR
    )

    c = np.asarray(c, dtype=np.float32) / 255.0

    rgb = (_jet(c) * 255).astype(np.uint8)

    alpha = (
        np.clip((c - 0.25) / 0.75, 0, 1) * 210
    ).astype(np.uint8)

    return Image.fromarray(
        np.dstack([rgb, alpha]),
        "RGBA"
    )


# ------------------------------------------------------------
# Image and DICOM reading
# ------------------------------------------------------------

def read_image(data: bytes, filename: str):
    """
    Read JPG, PNG or DICOM image.

    Returns:
        PIL RGB image
        is_dicom boolean
    """

    is_dicom = (
        filename.lower().endswith((".dcm", ".dicom"))
        or data[128:132] == b"DICM"
    )

    # Normal image
    if not is_dicom:

        return (
            Image.open(
                io.BytesIO(data)
            ).convert("RGB"),
            False
        )

    # DICOM image
    import pydicom

    ds = pydicom.dcmread(
        io.BytesIO(data)
    )

    arr = ds.pixel_array.astype(np.float32)

    # Handle RGB and multi-frame images
    if arr.ndim == 3:

        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]

        else:
            arr = arr[arr.shape[0] // 2]

    # Apply DICOM rescale parameters
    arr = (
        arr * float(getattr(ds, "RescaleSlope", 1))
        + float(getattr(ds, "RescaleIntercept", 0))
    )

    # Intensity normalization
    lo, hi = np.percentile(
        arr,
        (1, 99)
    )

    arr = np.clip(
        (arr - lo) / (hi - lo + 1e-8),
        0,
        1
    )

    # MONOCHROME1 inversion
    if (
        getattr(ds, "PhotometricInterpretation", "")
        == "MONOCHROME1"
    ):
        arr = 1 - arr

    # Convert to PIL image
    image = Image.fromarray(
        (arr * 255).astype(np.uint8)
    ).convert("RGB")

    return image, True
