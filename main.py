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

print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the content and style layers
content_layers = ["conv_4"]
style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

# Extract the activations from the chosen layers
vgg = m.StyleTransferModel()
vgg.vgg.to(device)


content_image = utils.load_image(r"E:\cov-project\woman.jpg", utils.transform)
style_image = utils.load_image(
    r"E:\cov-project\style_transfer\styles\studio_ghibli\ghibli_1.jpg",
    utils.transform,
    shape=content_image.shape[-2:],
)

target = content_image.clone().requires_grad_(True).to(device)
optimizer = optim.Adam([target], lr=0.003)
mse_loss = nn.MSELoss()

model = m.VGG(content_layers, style_layers, vgg.vgg).to(device).eval()

num_steps = 300
content_weight = 1
style_weight = 1e6

for step in range(num_steps):
    target_features = model(target)
    content_features = model(content_image)
    style_features = model(style_image)

    # Compute the content loss
    content_loss = content_weight * mse_loss(
        target_features["conv_4"], content_features["conv_4"]
    )

    # Compute the style loss
    style_loss = 0
    for layer in style_layers:
        target_gram = utils.gram_matrix(target_features[layer])
        style_gram = utils.gram_matrix(style_features[layer])
        layer_style_loss = mse_loss(target_gram, style_gram)
        _, d, h, w = target_features[layer].shape
        style_loss += layer_style_loss / (d * h * w)

    total_loss = content_loss + (style_weight * style_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(
            f"Step [{step}/{num_steps}], Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss:.4f}"
        )

final_img = target.clone().squeeze(0)
final_img = final_img.cpu().clamp_(0, 1)
transforms.ToPILImage()(final_img).save("output_image.jpg")

print("Gracefully finished")
