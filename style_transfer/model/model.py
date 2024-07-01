"""Module implementing the style transfer and normalization"""
import torch
import torchvision.models as models
from torch import nn
from torchvision.models import VGG19_Weights


class StyleTransferModel:
    """
    Class to initialize and return a pre-trained VGG19 model
    for style transfer.
    """

    def __init__(self, encoder: str) -> None:
        """
        Initialize the StyleTransferModel with a pre-trained VGG19 model.

        :param encoder: str
            Name of the pre-trained encoder we want to use.
        """
        if encoder == "vgg19":
            self.encoder = models.vgg19(
                weights=VGG19_Weights.IMAGENET1K_V1
            ).features.eval()
        elif encoder == "resnet50":
            resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.encoder = torch.nn.Sequential(*list(resnet50.children())[:-2]).eval()
        elif encoder == "inception_v3":
            inception_v3 = models.inception_v3(
                weights=models.Inception_V3_Weights.IMAGENET1K_V1
            )
            self.encoder = torch.nn.Sequential(
                *list(inception_v3.children())[:-2]
            ).eval()
        else:
            raise ValueError(f"Encoder {encoder} is not supported.")

    def get_model(self) -> nn.Module:
        """
        Return the pre-trained VGG19 model.

        :return: nn.Module
            The VGG19 feature extractor model.
        """
        print(self.encoder)
        return self.encoder


class Normalization(nn.Module):
    """
    Module to normalize an image tensor using the mean and standard deviation
    of the ImageNet dataset.
    """

    def __init__(self) -> None:
        """
        Initialize the Normalization module with mean and standard deviation.
        Mean and standard deviation values are from ImageNet standards.
        """
        super(Normalization, self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to normalize the image tensor.

        :param img: torch.Tensor
            The input image tensor with shape [B x C x H x W].
        :return: torch.Tensor
            The normalized image tensor.
        """
        return (img - self.mean) / self.std

    def get_mean(self) -> torch.Tensor:
        """
        Get the mean used for normalization.

        :return: torch.Tensor
            The mean tensor.
        """
        return self.mean

    def get_std(self) -> torch.Tensor:
        """
        Get the standard deviation used for normalization.

        :return: torch.Tensor
            The standard deviation tensor.
        """
        return self.std
