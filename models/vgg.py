import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from models.utils import make_layers, _initialize_weights


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes, pretrained, init_weights=True):
        super(VGG, self).__init__()
        self.model_name = 'vgg'
        self.pretrained = pretrained
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights and not pretrained:
            _initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


cfgs = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'B': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(cfg, batch_norm, progress, **kwargs):
    pretrained = kwargs['pretrained']

    model = VGG(make_layers(cfgs[cfg], conv_kernel_size=3, batch_norm=batch_norm), **kwargs)

    if pretrained:
        if batch_norm:
            model_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        else:
            model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        state_dict = load_state_dict_from_url(model_url,
                                              progress=progress)

        state_dict['features.0.weight'] = state_dict['features.0.weight'][:, 0, :, :].unsqueeze(1)
        state_dict = {param_name: param for param_name, param in state_dict.items() if not('classifier' in param_name)}

        model.load_state_dict(state_dict, strict=False)
    return model
