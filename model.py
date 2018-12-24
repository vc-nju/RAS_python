import torch
import torch.nn as nn
import torch.nn.functional as F


class RAS(nn.Module):
    def __init__(self):
        super(RAS, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv1_dsn6 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv2_dsn6 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.conv3_dsn6 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.conv4_dsn6 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.conv5_dsn6 = nn.Conv2d(256, 1, kernel_size=1)
        self.conv5_dsn6_up = nn.ConvTranspose2d(1, 1, kernel_size=64, stride=32)

        self.conv5_dsn6_5 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)
        self.conv1_dsn5 = nn.Conv2d(512, 64, kernel_size=1)
        self.conv2_dsn5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sum_dsn5_up = nn.ConvTranspose2d(1, 1, kernel_size=32, stride=16)

        self.sum_dsn5_4 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)
        self.conv1_dsn4 = nn.Conv2d(512, 64, kernel_size=1)
        self.conv2_dsn4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sum_dsn4_up = nn.ConvTranspose2d(1, 1, kernel_size=16, stride=8)

        self.sum_dsn4_3 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)
        self.conv1_dsn3 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv2_dsn3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sum_dsn3_up = nn.ConvTranspose2d(1, 1, kernel_size=8, stride=4)

        self.sum_dsn3_2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)
        self.conv1_dsn2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv2_dsn2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sum_dsn2_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2)

        self.conv1_dsn1 = nn.Conv2d(1, 64, kernel_size=1)
        self.conv2_dsn1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)


    def forward(self, x):
        x_size = x.size()
        x = F.relu(self.conv1_1(x))
        conv1_2 = F.relu(self.conv1_2(x))
        x = F.max_pool2d(conv1_2, kernel_size=2, stride=2)
        x = F.relu(self.conv2_1(x))
        conv2_2 = F.relu(self.conv2_2(x))
        x = F.max_pool2d(conv2_2, kernel_size=2, stride=2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        conv3_3 = F.relu(self.conv3_3(x))
        x = F.max_pool2d(conv3_3, kernel_size=2, stride=2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        conv4_3 = F.relu(self.conv4_3(x))
        x = F.max_pool2d(conv4_3, kernel_size=3, stride=2, padding=1)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        conv5_3 = F.relu(self.conv5_3(x))
        x = F.max_pool2d(conv5_3, kernel_size=3, stride=2, padding=1)
        
        x = self.conv1_dsn6(x)
        x = F.relu(self.conv2_dsn6(x))
        x = F.relu(self.conv3_dsn6(x))
        x = F.relu(self.conv4_dsn6(x))
        x = self.conv5_dsn6(x)
        upscore_dsn6 = self.crop(self.conv5_dsn6_up(x), x_size)
        
        x = self.conv5_dsn6_5(x)
        crop1_dsn5 = self.crop(x, conv5_3.size())
        sigmoid_dsn5 = nn.Sigmoid()
        x = -1*(sigmoid_dsn5(crop1_dsn5))+1
        x = x.expand(-1, 512, -1, -1).mul(conv5_3)
        x = self.conv1_dsn5(x)
        x = F.relu(self.conv2_dsn5(x))
        x = F.relu(self.conv3_dsn5(x))
        x = self.conv4_dsn5(x).sum(crop1_dsn5)
        upscore_dsn5 = self.crop(self.sum_dsn5_up(x), x_size)

        x = self.sum_dsn5_4(x)
        crop1_dsn4 = self.crop(x, conv4_3.size())
        sigmoid_dsn4 = nn.Sigmoid()
        x = -1*(sigmoid_dsn4(crop1_dsn4))+1
        x = x.expand(-1, 512, -1, -1).mul(conv4_3)
        x = self.conv1_dsn4(x)
        x = F.relu(self.conv2_dsn4(x))
        x = F.relu(self.conv3_dsn4(x))
        x = self.conv4_dsn4(x).sum(crop1_dsn4)
        upscore_dsn4 = self.crop(self.sum_dsn4_up(x), x_size)

        x = self.sum_dsn4_3(x)
        crop1_dsn3 = self.crop(x, conv3_3.size())
        sigmoid_dsn3 = nn.Sigmoid()
        x = -1*(sigmoid_dsn3(crop1_dsn3))+1
        x = x.expand(-1, 256, -1, -1).mul(conv3_3)
        x = self.conv1_dsn3(x)
        x = F.relu(self.conv2_dsn3(x))
        x = F.relu(self.conv3_dsn3(x))
        x = self.conv4_dsn3(x).sum(crop1_dsn3)
        upscore_dsn3 = self.crop(self.sum_dsn3_up(x), x_size)


        x = self.sum_dsn3_2(x)
        crop1_dsn2 = self.crop(x, conv2_2.size())
        sigmoid_dsn2 = nn.Sigmoid()
        x = -1*(sigmoid_dsn2(crop1_dsn2))+1
        x = x.expand(-1, 128, -1, -1).mul(conv2_2)
        x = self.conv1_dsn2(x)
        x = F.relu(self.conv2_dsn3(x))
        x = F.relu(self.conv2_dsn3(x))
        x = self.conv4_dsn2(x).sum(crop1_dsn2)
        upscore_dsn2 = self.crop(self.sum_dsn2_up(x), x_size)

        sigmoid_dsn1 = nn.Sigmoid()
        x = -1*(sigmoid_dsn1(upscore_dsn2))+1
        x = x.expand(-1, 64, -1, -1).mul(conv1_2)
        x = self.conv1_dsn1(x)
        x = F.relu(self.conv2_dsn1(x))
        x = F.relu(self.conv3_dsn1(x))
        x = self.conv4_dsn1(x).sum(upscore_dsn2)
        upscore_dsn1 = self.crop(self.sum_dsn2_up(x), x_size)

        return upscore_dsn1, upscore_dsn2, upscore_dsn3, upscore_dsn4, upscore_dsn5, upscore_dsn6 

    def crop(self, upsampled, x_size):
        c = (upsampled.size()[2] - x_size[2]) // 2
        assert(c>0)
        return upsampled[:, :, c:-c, c:-c]