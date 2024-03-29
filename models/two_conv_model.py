from torch import nn, flatten
from models.utils import make_layers


class TwoConv(nn.Module):
    def __init__(self, features, batch_norm, num_classes=10):
        super(TwoConv, self).__init__()
        self.model_name = 'two_conv'
        if batch_norm:
            self.model_name += '_batch_norm'
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


def two_conv(cfg, **kwargs):
    batch_norm = kwargs['batch_norm']
    model = TwoConv(make_layers(cfgs[cfg], conv_kernel_size=5, batch_norm=batch_norm), **kwargs)
    return model

