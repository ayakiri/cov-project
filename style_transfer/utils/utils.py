from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import tensor
from torchvision.transforms import transforms


def load_image(image_path: str, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [transforms.Resize((512, 768)), transforms.ToTensor()]
    )

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float32)


def save_image(image: torch.Tensor, output_path: str) -> None:
    image_np = image.squeeze(0).permute(1, 2, 0).numpy()
    plt.imsave(output_path, image_np)
