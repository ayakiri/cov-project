"""Module containing various utility functions for model-based style transfer."""
from typing import List, Tuple

import torch
from torch import Tensor, nn, optim

from style_transfer.model.loss import ContentLoss, StyleLoss
from style_transfer.model.model import Normalization


def build_model(
    cnn: nn.Module, style_img: torch.Tensor, content_img: torch.Tensor
) -> Tuple[nn.Sequential, List[StyleLoss], List[ContentLoss]]:
    """
    Build the style transfer model and compute the style and content losses.

    :param cnn: nn.Module
        The pre-trained CNN model.
    :param style_img: torch.Tensor
        The style image tensor.
    :param content_img: torch.Tensor
        The content image tensor.
    :return: tuple
        A tuple containing the style transfer model, a list of style loss modules,
        and a list of content loss modules.
    """
    normalization = Normalization()
    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    model = nn.Sequential(normalization)
    content_losses: List = []
    style_losses: List = []

    i = 0
    for layer in cnn.children():
        name, layer = layer_name(layer, i)
        if name.startswith("conv"):
            i += 1

        model.add_module(name, layer)
        add_loss_layers(
            model,
            content_img,
            content_layers,
            style_img,
            style_layers,
            content_losses,
            style_losses,
            name,
        )

    print("Model Architecture:")
    print(model)

    return finalize_model(model, content_losses, style_losses)


def layer_name(layer: nn.Module, i: int) -> Tuple[str, nn.Module]:
    """
    Generate a name for a given layer and return the layer with its name.

    :param layer: nn.Module
        The layer to be named.
    :param i: int
        The index for the convolutional layer.
    :return: tuple
        A tuple containing the name of the layer and the layer itself.
    """
    if isinstance(layer, nn.Conv2d):
        return f"conv_{i+1}", layer
    elif isinstance(layer, nn.ReLU):
        return f"relu_{i+1}", nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        return f"pool_{i+1}", layer
    elif isinstance(layer, nn.BatchNorm2d):
        return f"bn_{i+1}", layer
    else:
        raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")


def add_loss_layers(
    model: nn.Sequential,
    content_img: torch.Tensor,
    content_layers: List[str],
    style_img: torch.Tensor,
    style_layers: List[str],
    content_losses: List[ContentLoss],
    style_losses: List[StyleLoss],
    name: str,
) -> None:
    """
    Add content and style loss layers to the model.

    :param model: nn.Sequential
        The style transfer model being built.
    :param content_img: torch.Tensor
        The content image tensor.
    :param content_layers: list of str
        The names of layers to use for content loss.
    :param style_img: torch.Tensor
        The style image tensor.
    :param style_layers: list of str
        The names of layers to use for style loss.
    :param content_losses: list of ContentLoss
        The list to store content loss modules.
    :param style_losses: list of StyleLoss
        The list to store style loss modules.
    :param name: str
        The name of the current layer.
    """
    if name in content_layers:
        target = model(content_img).detach()
        content_loss = ContentLoss(target)
        model.add_module(f"content_loss_{name.split('_')[1]}", content_loss)
        content_losses.append(content_loss)

    if name in style_layers:
        target_feature = model(style_img).detach()
        style_loss = StyleLoss(target_feature)
        model.add_module(f"style_loss_{name.split('_')[1]}", style_loss)
        style_losses.append(style_loss)


def finalize_model(
    model: nn.Sequential,
    content_losses: List[ContentLoss],
    style_losses: List[StyleLoss],
) -> Tuple[nn.Sequential, List[StyleLoss], List[ContentLoss]]:
    """
    Finalize the model by trimming layers after the last loss layer.

    :param model: nn.Sequential
        The style transfer model being built.
    :param content_losses: list of ContentLoss
        The list of content loss modules.
    :param style_losses: list of StyleLoss
        The list of style loss modules.
    :return: tuple
        A tuple containing the finalized model, style loss modules, and content loss modules.
    """
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[: i + 1]
    return model, style_losses, content_losses


def compute_losses(
    style_losses: List[StyleLoss],
    content_losses: List[ContentLoss],
    style_weight: int,
    content_weight: int,
) -> tuple[Tensor, Tensor]:
    """
    Compute the style and content losses.

    :param style_losses: list of StyleLoss
        The style loss modules.
    :param content_losses: list of ContentLoss
        The content loss modules.
    :param style_weight: int
        The weight for the style loss.
    :param content_weight: int
        The weight for the content loss.
    :return: tuple
        A tuple containing the computed style score and content score.
    """
    style_score = torch.Tensor(sum(sl.loss for sl in style_losses) * style_weight)
    content_score = torch.Tensor(sum(cl.loss for cl in content_losses) * content_weight)
    return style_score, content_score


def style_transfer(
    model: nn.Module,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    input_img: torch.Tensor,
    optimizer_chosen: str,
    num_steps: int = 360,
    style_weight: int = 1000000,
    content_weight: int = 1,
) -> torch.Tensor:
    """
    Run the style transfer.

    :param model: nn.Module
        The style transfer model.
    :param content_img: torch.Tensor
        The content image tensor.
    :param style_img: torch.Tensor
        The style image tensor.
    :param input_img: torch.Tensor
        The input image tensor to optimize.
    :param optimizer_chosen: str
        The optimizer name used to update.
    :param num_steps: int, optional
        The number of optimization steps.
    :param style_weight: int, optional
        The weight for the style loss (default is 1000000).
    :param content_weight: int, optional
        The weight for the content loss (default is 1).
    :return: torch.Tensor
        The final stylized image tensor.
    """
    model, style_losses, content_losses = build_model(model, style_img, content_img)
    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    if optimizer_chosen == "lbfgs":
        optimizer = optim.LBFGS([input_img])
    # TODO - optimizer on param

    run = [0]
    while run[0] < num_steps:

        def closure() -> int:
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score, content_score = compute_losses(
                style_losses, content_losses, style_weight, content_weight
            )

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 10 == 0:
                print(
                    f"Steps: {run[0]} / {num_steps} - "
                    f"content loss: {content_score.item():.4f} style loss: {style_score.item():.4f}"
                )

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img
