"""Module for losses for style transfer operations"""
import torch
import torch.nn.functional as F
from torch import nn


class ContentLoss(nn.Module):
    """
    Module to compute content loss using mean squared error between
    the predicted feature maps and target feature maps.
    """

    def __init__(self, target: torch.Tensor) -> None:
        """
        Initialize the ContentLoss module.

        :param target: torch.Tensor
            The target feature maps that we want to match.
        """
        super(ContentLoss, self).__init__()
        self.loss = None
        self.target = target.detach()

    def forward(self, predict: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the content loss.

        :param predict: torch.Tensor
            The predicted feature maps.
        :return: torch.Tensor
            The input tensor is returned as-is.
        """
        self.loss = F.mse_loss(predict, self.target)
        return predict


class StyleLoss(nn.Module):
    """
    Module to compute style loss using the Gram matrix of the feature maps.
    The style loss is computed as the mean squared error between
    the Gram matrix of the predicted feature maps and the target feature maps.
    """

    def __init__(self, target_feature: torch.Tensor) -> None:
        """
        Initialize the StyleLoss module.

        :param target_feature: torch.Tensor
            The target feature maps whose Gram matrix is used to compute the style loss.
        """
        super(StyleLoss, self).__init__()
        self.loss = None
        self.target = gram_matrix(target_feature).detach()

    def forward(self, predict: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the style loss.

        :param predict: torch.Tensor
            The predicted feature maps.
        :return: torch.Tensor
            The input tensor is returned as-is.
        """
        gram_m = gram_matrix(predict)
        self.loss = F.mse_loss(gram_m, self.target)
        return predict


def gram_matrix(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gram matrix of a tensor input.

    The Gram matrix is computed as the dot product of the
    vectorized feature maps with their transposed vectors.

    :param input_tensor: torch.Tensor
        The input tensor with shape (a, b, c, d) where:
        - a is the batch size (usually 1)
        - b is the number of feature maps
        - c, d are the dimensions of a feature map
    :return: torch.Tensor
        The computed Gram matrix.
    """
    a, b, c, d = input_tensor.size()
    features = input_tensor.view(a * b, c * d)
    gram_m = torch.mm(features, features.t())
    return gram_m.div(a * b * c * d)
