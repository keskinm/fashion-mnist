from torch import nn, flatten
from models.utils import _initialize_weights, make_layers


class FiveConv(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=False):
        super(FiveConv, self).__init__()
        self.model_name = 'five_conv'
        self.features = features
        self.classifier = nn.Linear(512 * 3 * 3, num_classes)

        if init_weights:
            _initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = nn.MaxPool2d(kernel_size=9, stride=1)(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        return x


cfgs = {
    'A': [32, 64, 128, 'M', 256, 512, 'M'],
}


def five_conv(cfg, batch_norm, **kwargs):
    model = FiveConv(make_layers(cfg=cfgs[cfg], conv_kernel_size=6, batch_norm=batch_norm), **kwargs)
    return model

