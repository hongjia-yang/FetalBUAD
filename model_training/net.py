import torch
import torch.nn as nn
import torch.nn.functional as F


channel=[32,64,128,256]


class unet(nn.Module):
    def __init__(self, in_channel=1):
        super(unet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channel, channel[0], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[0]),
            nn.ReLU(),
            nn.Conv3d(channel[0], channel[0], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[0]),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(channel[0], channel[1], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[1]),
            nn.ReLU(),
            nn.Conv3d(channel[1], channel[1], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[1]),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv3d(channel[1], channel[2], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[2]),
            nn.ReLU(),
            nn.Conv3d(channel[2], channel[2], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[2]),
            nn.ReLU(),
        )
        self.encoder4 = nn.Sequential(
            nn.Conv3d(channel[2], channel[3], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[3]),
            nn.ReLU(),
            nn.Conv3d(channel[3], channel[3], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[3]),
            nn.ReLU(),
        )

        self.dropout1=nn.Dropout(0.3)

        self.upsample3 = nn.ConvTranspose3d(channel[3], channel[3], kernel_size=2, stride=2)     
        self.decoder3 = nn.Sequential(
            nn.Conv3d(channel[3]+channel[2], channel[2], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[2]),
            nn.ReLU(),
            nn.Conv3d(channel[2], channel[2], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[2]),
            nn.ReLU(),   
        )
        self.upsample2 = nn.ConvTranspose3d(channel[2], channel[2], kernel_size=2, stride=2)     
        self.decoder2 = nn.Sequential(
            nn.Conv3d(channel[2]+channel[1], channel[1], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[1]),
            nn.ReLU(),
            nn.Conv3d(channel[1], channel[1], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[1]),
            nn.ReLU(),   
        )
        self.upsample1 = nn.ConvTranspose3d(channel[1], channel[1], kernel_size=2, stride=2)     
        self.decoder1 = nn.Sequential(
            nn.Conv3d(channel[1]+channel[0], channel[0], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[0]),
            nn.ReLU(),
            nn.Conv3d(channel[0], channel[0], 3, stride=1, padding=1),
            nn.BatchNorm3d(channel[0]),
            nn.ReLU(),   
        )

        self.outputconv_seg =   nn.Conv3d(channel[0], 9, 1, stride=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout=nn.Dropout(0.1)
        self.fc = nn.Linear(256, 1)

        self.outputconv_age =   nn.Conv3d(channel[0], 1, 1, stride=1, padding=0)


    def forward(self, x):
        out = self.encoder1(x)
        t1 = out
        out = F.max_pool3d(out,2,2)

        out = self.encoder2(out)
        t2 = out
        out = F.max_pool3d(out,2,2)

        out = self.encoder3(out)
        t3 = out
        out = F.max_pool3d(out,2,2)

        out = self.encoder4(out)        
        t4 = out

        out=self.dropout1(out)

        age=self.avgpool(out)
        age = age.view(age.size(0), -1)
        age=self.dropout(age)
        age=self.fc(age)

        out=self.upsample3(out)
        out=torch.cat((out,t3),dim=1)
        out=self.decoder3(out)

        out=self.upsample2(out)
        out=torch.cat((out,t2),dim=1)
        out=self.decoder2(out)

        out=self.upsample1(out)
        out=torch.cat((out,t1),dim=1)
        out=self.decoder1(out)

        out_seg=self.outputconv_seg(out)
        out_age=self.outputconv_age(out)

        return age,out_seg,out_age

