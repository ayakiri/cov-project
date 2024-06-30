"""Module containing utility functions for image processing."""
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import transforms


def load_image(image_path: str, device: torch.device) -> torch.Tensor:
    """
    Load an image from the given path and apply transformations to resize and convert it to a tensor.

    :param image_path: str
        The path to the image file.
    :param device: torch.device
        The device to which the image tensor should be moved.
    :return: torch.Tensor
        The transformed image tensor.
    """
    transform = transforms.Compose(
        [transforms.Resize((480, 640)), transforms.ToTensor()]
    )

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float32)


def save_image(image: torch.Tensor, output_path: str) -> None:
    """
    Save a tensor as an image to the given path.

    :param image: torch.Tensor
        The image tensor to be saved. Expected shape is [1, C, H, W].
    :param output_path: str
        The path where the image file should be saved.
    """
    image_np = image.squeeze(0).permute(1, 2, 0).detach().numpy()
    plt.imsave(output_path, image_np)
