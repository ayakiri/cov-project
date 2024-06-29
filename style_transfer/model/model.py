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
