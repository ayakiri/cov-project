import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import transforms


def save_image(content_image: torch.Tensor, output_path: str) -> None:
    print(type(content_image))
    content_image_np = content_image.permute(1, 2, 0).numpy()
    plt.imsave(output_path, content_image_np)


def load_content_img(path: str) -> torch.Tensor:
    img = Image.open(path)
    transform = transforms.Compose(
        [
            # transforms.Resize((512, 512)),
            transforms.ToTensor()
        ]
    )
    img_in_tensor = transform(img)
    print(img_in_tensor)
    return img_in_tensor
