import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms

import style_transfer.model.model as m
from style_transfer.utils import utils

parser = argparse.ArgumentParser(
    prog="Style Transfer", description="AI style transfer model"
)

parser.add_argument("-f", "--file-path", type=str, required=True, help="Path to image")
parser.add_argument(
    "-o", "--output-path", type=str, required=True, help="Path to save image"
)
parser.add_argument(
    "-s",
    "--style",
    type=str,
    required=True,
    help="Style you want",
    choices=["ghibli", "pop_art"],
)
args = parser.parse_args()

if args.style == "ghibli":
    style_image_path = r"style_transfer/styles/studio_ghibli/ghibli_1.jpg"
elif args.style == "pop_art":
    style_image_path = r"style_transfer/styles/pop_art/pop_art.jpg"
else:
    style_image_path = r"style_transfer/styles/studio_ghibli/ghibli_1.jpg"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

print(args, device, type(device))

content_image = utils.load_image(r"E:\cov-project\woman.jpg", device)
style_image = utils.load_image(
    r"E:\cov-project\style_transfer\styles\studio_ghibli\ghibli_1.jpg", device
)

utils.save_image(content_image, args.output_path)

print("Gracefully finished")
