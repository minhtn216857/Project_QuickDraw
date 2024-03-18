import torch
import torch.nn as nn
from torchvision.transforms import Compose

class Module_QuickDraw(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.layer1 = self._conv_box(1, 16, 3, 1)
        self.layer2 = self._conv_box(16, 64, 3, 1)
        self.layer3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
        )
        self.layer4 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=256),
        )
        self.layer5 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    def _conv_box(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

if __name__ == '__main__':
    test_img = torch.rand(8, 1, 28, 28)
    print(test_img.shape)
    module = Module_QuickDraw()
    test_img = module(test_img)
    print(test_img.shape)