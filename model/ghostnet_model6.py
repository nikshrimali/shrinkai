from data_process.ghost_batchnorm import *
from torch import nn
from torch.functional import F


class GhostNet(nn.Module):

    def __init__(self):
        super(GhostNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 3,padding=0),  # 26
            nn.ReLU(),
            GhostBatchNorm(num_features=10, num_splits=4),
            nn.Dropout(0.03)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0),  # 24
            nn.ReLU(),
            GhostBatchNorm(num_features=16, num_splits=4),
            nn.Dropout(0.03),
            nn.MaxPool2d(2,2), # 12
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 10, 3, padding=0),  # 10
            nn.ReLU(),
            GhostBatchNorm(num_features=10, num_splits=4),
            nn.Dropout(0.03)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0),  # 8
            nn.ReLU(),
            GhostBatchNorm(num_features=16, num_splits=4),
            nn.Dropout(0.03),
            # nn.MaxPool2d(2,2),  # 7
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 10, 3),  # 6
            nn.ReLU(),
            GhostBatchNorm(num_features=10, num_splits=4),
            nn.Dropout(0.03)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(10, 10, 3),  # 4
            nn.ReLU(),
            GhostBatchNorm(num_features=10, num_splits=4),
            nn.Dropout(0.03)
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(10, 10, 1),  #1
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = self.conv7(x)
        x = x.view(-1,10)
        return F.log_softmax(x, dim=-1)
