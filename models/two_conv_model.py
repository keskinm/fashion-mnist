from torch import nn, flatten
from models.utils import make_layers


class TwoConv(nn.Module):
    def __init__(self, features, num_classes=10):
        super(TwoConv, self).__init__()
        self.features = features
        self.fc = nn.Linear(22 * 22 * 32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = flatten(x, 1)
        x = self.fc(x)
        return x


cfgs = {
    'A': [16, 'M', 32, 'M'],
}


def _two_conv(cfg, batch_norm, **kwargs):
    model = TwoConv(make_layers(cfgs[cfg], conv_kernel_size=5, batch_norm=batch_norm), **kwargs)
    return model


def two_conv(**kwargs):
    return _two_conv('A', batch_norm=False, **kwargs)


def two_conv_bn(**kwargs):
    return _two_conv('A', batch_norm=True, **kwargs)
