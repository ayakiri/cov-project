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
    choices=["ghibli", "pop_art"],
)
args = parser.parse_args()

if args.style == "ghibli":
    style_image_path = r"style_transfer/styles/studio_ghibli/ghibli_1.jpg"
elif args.style == "pop_art":
    style_image_path = r"style_transfer/styles/pop_art/pop_art.jpg"
else:
    style_image_path = r"style_transfer/styles/studio_ghibli/ghibli_1.jpg"


print(args)

# Model create
model = m.ModelClass()
model.model_summary()

# Content image prepare
content_image = utils.load_image(args.file_path)
# utils.save_image(content_image, args.output_path)
content_image = utils.add_batch_dim(content_image)
utils.image_info(content_image)

# Style image prepare
style_image = utils.load_image(style_image_path)
style_image = utils.add_batch_dim(style_image)
utils.image_info(style_image)

print("Done")
