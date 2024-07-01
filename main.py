import argparse

import torch

from style_transfer.model.model import Normalization, StyleTransferModel
from style_transfer.utils import image_utils, model_utils

parser = argparse.ArgumentParser(
    prog="Style Transfer", description="AI style transfer model"
)

parser.add_argument("-fp", "--file-path", type=str, required=True, help="Path to image")
parser.add_argument(
    "-op", "--output-path", type=str, required=True, help="Path to save image"
)
parser.add_argument(
    "-sp", "--style-path", type=str, required=True, help="Style you want"
)
parser.add_argument("-s", "--steps", type=int, default=360, help="How many steps")
parser.add_argument(
    "-e",
    "--encoder",
    type=str,
    default="vgg19",
    help="Which encoder to use",
    choices=["vgg19", "resnet50", "inception_v3"],
)
parser.add_argument(
    "-o",
    "--optimizer",
    type=str,
    default="lbfgs",
    help="Which optimizer to use",
    choices=["lbfgs"],
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

print(args, device, type(device))

content_image = image_utils.load_image(args.file_path, device)
style_image = image_utils.load_image(args.style_path, device)
input_image = content_image.clone()

model = StyleTransferModel(encoder=args.encoder).get_model()
normalization = Normalization()

output = model_utils.style_transfer(
    model=model,
    content_img=content_image,
    style_img=style_image,
    input_img=input_image,
    num_steps=args.steps,
    optimizer=args.optimizer,
)

print("Saving new image...")
image_utils.save_image(output, args.output_path)

print("Gracefully finished")
