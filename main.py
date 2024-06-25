import argparse

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
    choices=["one", "two"],
)
args = parser.parse_args()

print(args)

model = m.ModelClass()

model.model_summary()

img = utils.load_content_img(args.file_path)

utils.save_image(img, args.output_path)

print("Done")
