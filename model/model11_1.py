import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        '''Creates a basic block'''
        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()


    def forward(self, x, layer):
        # Normal Connection
        out = F.relu(self.dropout(self.bn1(self.conv1(x))))
        out = self.dropout(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DavidNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(DavidNet, self).__init__()
        self.in_planes = 64

        # Block 1
        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )

        # Block 2
        self.add_layer2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                )
        self.resnetlayer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        # Block 3
        self.resnetlayer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        # Block 4
        self.add_layer4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                )
        self.resnetlayer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        
        for stride in strides:
            print(f'Inside for look Stride is {stride}')
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        # Block1
        prep_layer = self.prep_layer(x)
        y = self.add_layer2(prep_layer)

        # Block2
        out = self.resnetlayer2(y)
        out = out + y
        
        # Block3
        out = self.resnetlayer3(out)

        # Block4
        y = self.add_layer4(out)
        out = self.resnetlayer4(out)
        out = out + y

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)


def davidnet():
    return DavidNet(ResnetBlock, [2, 2, 2, 2])
