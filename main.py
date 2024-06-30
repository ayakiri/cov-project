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
    "--style-path",
    type=str,
    required=True,
    help="Style you want",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

print(args, device, type(device))

content_image = image_utils.load_image(args.file_path, device)
style_image = image_utils.load_image(args.style_path, device)
input_image = content_image.clone()

model = StyleTransferModel().get_model()
normalization = Normalization()

output = model_utils.style_transfer(model, content_image, style_image, input_image)

print("Saving new image...")
image_utils.save_image(output, args.output_path)

print("Gracefully finished")
