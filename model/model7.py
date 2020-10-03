import torch
from torch import nn
from torch.functional import F
DROPOUT = 0.1

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3,padding=2, padding_mode='replicate'),  # 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(DROPOUT),

            nn.Conv2d(32, 64, 3, padding=2, padding_mode='replicate'),  # 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.03),
            nn.MaxPool2d(2,2, padding=1), # 16
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, padding_mode='replicate'),  # 16
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.03),
            # nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=2, padding_mode='replicate'),  # 16
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),
            nn.MaxPool2d(2,2), # 8
        )



        self.dpthwise_sep3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, groups=128, padding=1, padding_mode='replicate'),  # 8
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),

            nn.Conv2d(128, 128, 1, groups=128, padding_mode='replicate'),  # 8
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),
        )

        self.norm_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='replicate'),  # 8
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),
            # nn.MaxPool2d(2,2),  # 7
        )


        self.dilation = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=2, dilation=2, padding_mode='replicate'),  # 8
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),
            # nn.MaxPool2d(2,2),  # 7
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 128, 1, padding_mode='replicate'),  # 10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),

            nn.Conv2d(128, 128, 3, padding=1, padding_mode='replicate'),  # 10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),
            nn.MaxPool2d(2,2),  # 7

            nn.Conv2d(128, 128, 3, padding=2, padding_mode='replicate'),  # 10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),
        )

        self.block5 = nn.Sequential(

            nn.AvgPool2d(kernel_size=7),
            nn.Conv2d(128, 10, 1),  #1

        )

    def get_seperable(self, x, sep= True):
        if sep:
            # print(x.shape)
            x.view(x.size(0), -1)
            x = self.norm_conv(x)
            # print('After reshape:',x.shape)
            return x
        else:
            # print(x.shape)
            x.view(x.size(0), -1)
            x = self.dilation(x)
            # print('After reshape:',x.shape)
            return x




    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.dpthwise_sep3(x)
        # y.view(y.size(0), -1)
        # print('before dil',x.shape)
        x = torch.cat((self.get_seperable(x, sep=True), self.dilation(x)),1)
        # print('after dil',x.shape)
        x = self.block4(x)
        x = self.block5(x)
        
        x = x.view(-1,10)
        return F.log_softmax(x, dim=-1)
