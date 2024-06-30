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

    def __init__(self) -> None:
        """
        Initialize the StyleTransferModel with a pre-trained VGG19 model.
        """
        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        # TODO - encoder on param

    def get_model(self) -> nn.Module:
        """
        Return the pre-trained VGG19 model.

        :return: nn.Module
            The VGG19 feature extractor model.
        """
        return self.vgg


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
