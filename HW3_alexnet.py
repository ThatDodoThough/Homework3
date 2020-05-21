import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
from gradient_reversal_example import ReverseLayerF

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.dann_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, alpha=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if alpha is None:   # normal classification, Gy
            x = self.classifier(x)
        else:
            reverse_features = ReverseLayerF.apply(x, alpha)
            x = self.dann_classifier(reverse_features)
        return x


def alexnet(pretrained=False, progress=True, num_classes):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)

        # Use same starting weights for Gy and Gd
        #FC6
        for wy, wd in zip(model.classifier[1].weight.data, model.dann_classifier[1].weight.data):
            wd = wy
        for by, bd in zip(model.classifier[1].bias.data, model.dann_classifier[1].bias.data):
            bd = by
        #FC7
        for wy, wd in zip(model.classifier[4].weight.data, model.dann_classifier[4].weight.data):
            wd = wy
        for by, bd in zip(model.classifier[4].bias.data, model.dann_classifier[4].bias.data):
            bd = by

        # Change output layers of Gy and Gd
        model.classifier[6] = nn.Linear(4096, num_classes)
        model.dann_classifier[6] = nn.Linear(4096, 2)

    return model
