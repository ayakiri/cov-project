import torch
import torchvision.models as models
from torch import nn
from torchvision.models import VGG19_Weights


class StyleTransferModel:
    def __init__(self) -> None:
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def model_summary(self) -> None:
        print("=================")
        print("Model summary:")
        print(self.vgg)
        print("=================")


class VGG(nn.Module):
    def __init__(
        self,
        content_layers: list,
        style_layers: list,
        model: torch.nn.modules.container.Sequential,
    ) -> None:
        super(VGG, self).__init__()
        self.chosen_features = content_layers + style_layers
        self.model = model[:29]

    def forward(self, x: torch.Tensor) -> dict:
        print(type(x))
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if f"conv_{int(name) + 1}" in self.chosen_features:
                features[f"conv_{int(name) + 1}"] = x
        return features
