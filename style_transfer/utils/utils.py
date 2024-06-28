from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import tensor
from torchvision.transforms import transforms


def gram_matrix(t: torch.Tensor) -> torch.Tensor:
    _, d, h, w = t.size()
    t = t.view(d, h * w)
    gram = torch.mm(t, t.t())
    return gram


def load_image(
    image_path: str,
    transform: transforms.Compose = None,
    max_size: int = 400,
    shape: Optional[torch.Size] = None,
    device: str = "cpu",
) -> torch.Tensor:
    image = Image.open(image_path)
    if max(image.size) > max_size:
        size = max_size
        image = image.resize((size, int(size * image.size[1] / image.size[0])))

    if transform:
        image = transform(image).unsqueeze(0)

    return image.to(device)


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
