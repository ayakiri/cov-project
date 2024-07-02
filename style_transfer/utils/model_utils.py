from typing import List, Tuple

import torch
from torch import nn, optim

from style_transfer.model.loss import ContentLoss, StyleLoss
from style_transfer.model.model import Normalization


def build_vgg19_model(
    cnn: nn.Module, style_img: torch.Tensor, content_img: torch.Tensor
) -> Tuple[nn.Sequential, List[StyleLoss], List[ContentLoss]]:
    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    return build_general_model(
        cnn, style_img, content_img, content_layers, style_layers, "vgg19"
    )


def build_resnet50_model(
    cnn: nn.Module, style_img: torch.Tensor, content_img: torch.Tensor
) -> Tuple[nn.Sequential, List[StyleLoss], List[ContentLoss]]:
    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    return build_general_model(
        cnn, style_img, content_img, content_layers, style_layers, "resnet50"
    )


def build_inception_v3_model(
    cnn: nn.Module, style_img: torch.Tensor, content_img: torch.Tensor
) -> Tuple[nn.Sequential, List[StyleLoss], List[ContentLoss]]:
    content_layers = ["layer_8"]
    style_layers = ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5"]

    return build_general_model(
        cnn, style_img, content_img, content_layers, style_layers, "inception_v3"
    )


def build_general_model(
    cnn: nn.Module,
    style_img: torch.Tensor,
    content_img: torch.Tensor,
    content_layers: List[str],
    style_layers: List[str],
    encoder: str,
) -> Tuple[nn.Sequential, List[StyleLoss], List[ContentLoss]]:
    normalization = Normalization()
    model = nn.Sequential(normalization)
    content_losses: List[ContentLoss] = []
    style_losses: List[StyleLoss] = []

    i = 0
    for layer in cnn.children():
        name, layer = layer_name(layer, i, encoder)
        if name.startswith("conv"):
            i += 1
        elif name.startswith("layer") and encoder == "inception_v3":
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


def layer_name(layer: nn.Module, i: int, encoder: str) -> Tuple[str, nn.Module]:
    if isinstance(layer, nn.Conv2d):
        return f"conv_{i + 1}", layer
    elif isinstance(layer, nn.ReLU):
        return f"relu_{i + 1}", nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
        return f"pool_{i + 1}", layer
    elif isinstance(layer, nn.BatchNorm2d):
        return f"bn_{i + 1}", layer
    elif isinstance(layer, nn.Linear):
        if encoder == "inception_v3":
            return "fc", layer
    else:
        return f"layer_{i + 1}", layer

    return "", layer


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    style_score = sum(sl.loss for sl in style_losses) * style_weight
    content_score = sum(cl.loss for cl in content_losses) * content_weight
    return style_score, content_score


def style_transfer(
    model: nn.Module,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    input_img: torch.Tensor,
    optimizer_chosen: str,
    encoder: str,
    num_steps: int = 360,
    style_weight: int = 1000000,
    content_weight: int = 1,
) -> torch.Tensor:
    if encoder == "vgg19":
        build_model_func = build_vgg19_model
    elif encoder == "resnet50":
        build_model_func = build_resnet50_model
    elif encoder == "inception_v3":
        build_model_func = build_inception_v3_model
    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    model, style_losses, content_losses = build_model_func(
        model, style_img, content_img
    )
    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    if optimizer_chosen == "lbfgs":
        optimizer = optim.LBFGS([input_img])
    # TODO - optimizer on param

    run = [0]
    while run[0] < num_steps:

        def closure() -> float:
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
                    f"content loss: {content_score:.10f} style loss: {style_score:.10f}"
                )

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img
