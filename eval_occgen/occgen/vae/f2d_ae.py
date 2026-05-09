import torch
import torch.nn as nn
from torch_fidelity import FeatureExtractorBase
from torchvision import models


class InceptionV3AE(nn.Module):
    def __init__(self, pretrained=True):
        super(InceptionV3AE, self).__init__()
        self.encoder = models.vgg16(pretrained=pretrained)
        self.encoder.fc = nn.Identity()  # Remove the final classification layer

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1000, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=3, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        if not isinstance(x, torch.Tensor):
            x = x.logits
        x = x.view(x.size(0), 1000, 1, 1)
        x = self.decoder(x)
        return x

    def extract_features(self, x):
        x = self.encoder(x.float())
        return x,

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


class F2D(FeatureExtractorBase):
    def __init__(self, name='f2dae', features_list=('2048',), feature_extractor_weights_path=None, **kwargs):
        super().__init__(name, features_list)
        self.net = InceptionV3AE(pretrained=True)
        if feature_extractor_weights_path is not None:
            self.load_state_dict(torch.load(feature_extractor_weights_path)['model_state_dict'])

    def train_forward(self, x):
        return self.net(x)

    def forward(self, x):
        return self.net.extract_features(x)

    @staticmethod
    def get_provided_features_list():
        return ['2048']

    @staticmethod
    def get_default_feature_layer_for_metric(metric):
        return '2048'

    @staticmethod
    def can_be_compiled():
        return True

    @staticmethod
    def get_dummy_input_for_compile():
        return torch.rand([1, 3, 299, 299])
