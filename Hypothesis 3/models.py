import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

class ProjectionBottleneck(nn.Module):
    """Projects shallow features to match the deepest feature map dimensions for L2 Hint Loss."""
    def __init__(self, in_channels, out_channels, downsample_factor):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=downsample_factor, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.project(x)

class SelfDistillationResNet18(nn.Module):
    def __init__(self, num_classes=8, in_channels=3):
        super().__init__()
        # Load base ResNet18
        base_model = resnet18(weights=None)

        # Modify initial conv for 32x32 images (like CIFAR/BloodMNIST) instead of 224x224 ImageNet
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        # We skip the maxpool to preserve spatial resolution for small images

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifiers for the 4 exits
        self.classifier1 = nn.Linear(64, num_classes)
        self.classifier2 = nn.Linear(128, num_classes)
        self.classifier3 = nn.Linear(256, num_classes)
        self.classifier4 = nn.Linear(512, num_classes)
        self.bottleneck1 = ProjectionBottleneck(64, 512, downsample_factor=8)
        self.bottleneck2 = ProjectionBottleneck(128, 512, downsample_factor=4)
        self.bottleneck3 = ProjectionBottleneck(256, 512, downsample_factor=2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        # Exit 1
        f1 = self.layer1(x)
        out1 = self.classifier1(self.avgpool(f1).flatten(1))
        b1 = self.bottleneck1(f1)

        # Exit 2
        f2 = self.layer2(f1)
        out2 = self.classifier2(self.avgpool(f2).flatten(1))
        b2 = self.bottleneck2(f2)

        # Exit 3
        f3 = self.layer3(f2)
        out3 = self.classifier3(self.avgpool(f3).flatten(1))
        b3 = self.bottleneck3(f3)

        # Exit 4 (Deepest / Teacher)
        f4 = self.layer4(f3)
        out4 = self.classifier4(self.avgpool(f4).flatten(1))
        b4 = f4

        logits = [out1, out2, out3, out4]
        bottlenecks = [b1, b2, b3, b4]

        return logits, bottlenecks

    def early_exit_forward(self, x, exit_idx):
        """Used strictly for inference timing. Stops computation at the specified exit."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        if exit_idx == 0:
            return self.classifier1(self.avgpool(x).flatten(1))

        x = self.layer2(x)
        if exit_idx == 1:
            return self.classifier2(self.avgpool(x).flatten(1))

        x = self.layer3(x)
        if exit_idx == 2:
            return self.classifier3(self.avgpool(x).flatten(1))

        x = self.layer4(x)
        return self.classifier4(self.avgpool(x).flatten(1))

class BaselineResNet18(nn.Module):
    def __init__(self, num_classes=8, in_channels=3):
        super().__init__()
        base_model = resnet18(weights=None)

        # Match the SD model's initial layers for 32x32 images
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x).flatten(1)
        return self.fc(x)

class SelfDistillationResNet50(nn.Module):
    def __init__(self, num_classes=8, in_channels=3):
        super().__init__()
        base_model = resnet50(weights=None)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1, self.relu = base_model.bn1, base_model.relu
        self.layer1, self.layer2 = base_model.layer1, base_model.layer2
        self.layer3, self.layer4 = base_model.layer3, base_model.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ResNet-50 Channel Dimensions: 256, 512, 1024, 2048
        self.classifier1 = nn.Linear(256, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)
        self.classifier3 = nn.Linear(1024, num_classes)
        self.classifier4 = nn.Linear(2048, num_classes) 

        self.bottleneck1 = ProjectionBottleneck(256, 2048, downsample_factor=8)
        self.bottleneck2 = ProjectionBottleneck(512, 2048, downsample_factor=4)
        self.bottleneck3 = ProjectionBottleneck(1024, 2048, downsample_factor=2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(x); out1 = self.classifier1(self.avgpool(f1).flatten(1)); b1 = self.bottleneck1(f1)
        f2 = self.layer2(f1); out2 = self.classifier2(self.avgpool(f2).flatten(1)); b2 = self.bottleneck2(f2)
        f3 = self.layer3(f2); out3 = self.classifier3(self.avgpool(f3).flatten(1)); b3 = self.bottleneck3(f3)
        f4 = self.layer4(f3); out4 = self.classifier4(self.avgpool(f4).flatten(1))
        return [out1, out2, out3, out4], [b1, b2, b3, f4]

    def early_exit_forward(self, x, exit_idx):
        x = self.relu(self.bn1(self.conv1(x)))
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        classifiers = [self.classifier1, self.classifier2, self.classifier3, self.classifier4]
        for i in range(exit_idx + 1):
            x = layers[i](x)
        return classifiers[exit_idx](self.avgpool(x).flatten(1))

class BaselineResNet50(nn.Module):
    def __init__(self, num_classes=8, in_channels=3):
        super().__init__()
        base_model = resnet50(weights=None)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1, self.relu = base_model.bn1, base_model.relu
        self.layer1, self.layer2 = base_model.layer1, base_model.layer2
        self.layer3, self.layer4 = base_model.layer3, base_model.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.avgpool(x).flatten(1))

class SelfDistillationResNet18_H10k(nn.Module):
    def __init__(self, num_classes=7, in_channels=3):
        super().__init__()
        base_model = resnet18(weights=None)

        # Standard ResNet18 initial layers for 224x224 images
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool

        # Classifiers for the 4 exits
        self.classifier1 = nn.Linear(64, num_classes)
        self.classifier2 = nn.Linear(128, num_classes)
        self.classifier3 = nn.Linear(256, num_classes)
        self.classifier4 = nn.Linear(512, num_classes) # Deepest (Teacher)

        self.bottleneck1 = ProjectionBottleneck(64, 512, downsample_factor=8)
        self.bottleneck2 = ProjectionBottleneck(128, 512, downsample_factor=4)
        self.bottleneck3 = ProjectionBottleneck(256, 512, downsample_factor=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # Exit 1
        f1 = self.layer1(x)
        out1 = self.classifier1(self.avgpool(f1).flatten(1))
        b1 = self.bottleneck1(f1)

        # Exit 2
        f2 = self.layer2(f1)
        out2 = self.classifier2(self.avgpool(f2).flatten(1))
        b2 = self.bottleneck2(f2)

        # Exit 3
        f3 = self.layer3(f2)
        out3 = self.classifier3(self.avgpool(f3).flatten(1))
        b3 = self.bottleneck3(f3)

        # Exit 4 (Deepest / Teacher)
        f4 = self.layer4(f3)
        out4 = self.classifier4(self.avgpool(f4).flatten(1))
        b4 = f4

        logits = [out1, out2, out3, out4]
        bottlenecks = [b1, b2, b3, b4]

        return logits, bottlenecks

    def early_exit_forward(self, x, exit_idx):
        """Used strictly for inference timing. Stops computation at the specified exit."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        
        x = self.layer1(x)
        if exit_idx == 0:
            return self.classifier1(self.avgpool(x).flatten(1))

        x = self.layer2(x)
        if exit_idx == 1:
            return self.classifier2(self.avgpool(x).flatten(1))

        x = self.layer3(x)
        if exit_idx == 2:
            return self.classifier3(self.avgpool(x).flatten(1))

        x = self.layer4(x)
        return self.classifier4(self.avgpool(x).flatten(1))

class Baseline_Resnet18_H10k(nn.Module):
    def __init__(self, num_classes=7, in_channels=3):
        super().__init__()
        base_model = resnet18(weights=None)

        # Standard initial layers for 224x224
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x).flatten(1)
        return self.fc(x)