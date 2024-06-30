import argparse

import torch

from style_transfer.model.model import Normalization, StyleTransferModel
from style_transfer.utils import image_utils, model_utils

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
    choices=["ghibli", "pop_art", "japanese"],
)
args = parser.parse_args()

if args.style == "ghibli":
    style_image_path = r"style_transfer/styles/studio_ghibli/ghibli_1.jpg"
elif args.style == "pop_art":
    style_image_path = r"style_transfer/styles/pop_art/pop_art.jpg"
elif args.style == "japanese":
    style_image_path = r"style_transfer/styles/japanese/japanese.jpg"
else:
    style_image_path = r"style_transfer/styles/studio_ghibli/ghibli_1.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

print(args, device, type(device))

content_image = image_utils.load_image(args.file_path, device)
style_image = image_utils.load_image(style_image_path, device)
input_image = content_image.clone()

model = StyleTransferModel().get_model()
normalization = Normalization()

# TODO - run model

print("Gracefully finished")
