from torch import nn, flatten
from models.utils import _initialize_weights, make_layers


class SevenConv(nn.Module):
    def __init__(self, features, num_classes, init_weights=True):
        super(SevenConv, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
        )

        if init_weights:
            _initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        return x


cfgs = {
    'A': [32, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 'M'],
}


def _seven_conv(cfg, batch_norm, **kwargs):
    model = SevenConv(make_layers(cfgs[cfg], conv_kernel_size=5, batch_norm=batch_norm), **kwargs)
    return model


def seven_conv(**kwargs):
    return _seven_conv('A', batch_norm=False, **kwargs)


def seven_conv_bn(**kwargs):
    return _seven_conv('A', batch_norm=True, **kwargs)
