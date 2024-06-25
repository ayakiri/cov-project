import torch
import torchvision.models as models


class ModelClass(torch.nn.Module):
    def __init__(self) -> None:
        super(ModelClass, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.vgg19 = torch.nn.Sequential(*list(vgg19.children())[:-1])
        self.features = torch.nn.Sequential(*list(vgg19.features.children()))

    def model_summary(self) -> None:
        print("Model summary:")
        print(self.features)
        print("=================")
        print(self.vgg19)
        print("=================")
        print(len(self.features))
