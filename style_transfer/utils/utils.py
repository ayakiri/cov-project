import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import tensor
from torchvision.transforms import transforms


# Image utils
def save_image(image: torch.Tensor, output_path: str) -> None:
    image_np = image.permute(1, 2, 0).numpy()
    plt.imsave(output_path, image_np)


def image_info(image: torch.Tensor) -> None:
    print("Image shape:", image.shape)
    print("Image max value:", image.min())
    print("Image min value:", image.max())


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path)
    transform = transforms.Compose(
        [
            # transforms.Resize((512, 512)), do we want that
            transforms.ToTensor()
        ]
    )
    img_in_tensor = transform(img)
    print(img_in_tensor)
    return img_in_tensor


# Ops utils
def add_batch_dim(tensor_3dim: torch.Tensor) -> torch.Tensor:
    """
    Adds batch to a tensor and returns a new tensor in (batch_size, channels, height, width) format.

    :param tensor_3dim: torch.Tensor
    :return: torch.Tensor
    """
    new_tensor = tensor_3dim.unsqueeze(0)
    return new_tensor
