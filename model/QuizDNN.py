import torch
from torch import nn
from torch.functional import F
DROPOUT = 0.1
'''
x1 = Input
x2 = Conv(x1)
x3 = Conv(x1 + x2)
x4 = MaxPooling(x1 + x2 + x3)
x5 = Conv(x4)
x6 = Conv(x4 + x5)
x7 = Conv(x4 + x5 + x6)
x8 = MaxPooling(x5 + x6 + x7)
x9 = Conv(x8)
x10 = Conv (x8 + x9)
x11 = Conv (x8 + x9 + x10)
x12 = GAP(x11)
x13 = FC(x12)'''

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3,padding=1, padding_mode='replicate'),  # 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(DROPOUT))

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'),  # 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.03)
            )
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'),  # 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.03))

        self.block4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'),  # 16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.03)
            )
        
        self.block5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'),  # 16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.03)
            )

        self.block6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'),  # 16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.03)
            )
        
        self.block7 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'),  # 8
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.03)
            )

        self.block8 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'),  # 8
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.03)
            )

        self.block9 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'),  # 8
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.03)
            )
        

        self.block10 = nn.Sequential(

            nn.AvgPool2d(kernel_size=8),
            nn.Conv2d(32, 10, 1),  #1

        )


    def forward(self,x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x1+x2)
        x4 = x1+x2+x3
        x4 = self.pool(x4)
        x5 = self.block4(x4)
        x6 = self.block5(x4+x5)
        x7 = self.block6(x4 + x5 + x6)
        x8 = x5 + x6 + x7
        x8 = self.pool(x8)
        x9 = self.block7(x8)
        x10 = self.block8(x8 + x9)
        x11 = self.block9(x8 + x9 + x10)
        x12 = self.block10(x11)
        x = x12.view(-1,10)
        return F.log_softmax(x, dim=-1)
