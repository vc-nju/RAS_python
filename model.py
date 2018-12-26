import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

LR_RATE = 0.0001


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
        self.conv5_dsn6_up = nn.ConvTranspose2d(
            1, 1, kernel_size=64, stride=32)

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

        self.conv1_dsn1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv2_dsn1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3_dsn1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_dsn1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.optim = optim.Adam(self.parameters(), lr=LR_RATE)
        self.apply(RAS.weights_init)

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
        x = -1*(torch.sigmoid(crop1_dsn5))+1
        x = x.expand(-1, 512, -1, -1).mul(conv5_3)
        x = self.conv1_dsn5(x)
        x = F.relu(self.conv2_dsn5(x))
        x = F.relu(self.conv3_dsn5(x))
        x = self.conv4_dsn5(x) + crop1_dsn5
        upscore_dsn5 = self.crop(self.sum_dsn5_up(x), x_size)

        x = self.sum_dsn5_4(x)
        crop1_dsn4 = self.crop(x, conv4_3.size())
        x = -1*(torch.sigmoid(crop1_dsn4))+1
        x = x.expand(-1, 512, -1, -1).mul(conv4_3)
        x = self.conv1_dsn4(x)
        x = F.relu(self.conv2_dsn4(x))
        x = F.relu(self.conv3_dsn4(x))
        x = self.conv4_dsn4(x) + crop1_dsn4
        upscore_dsn4 = self.crop(self.sum_dsn4_up(x), x_size)

        x = self.sum_dsn4_3(x)
        crop1_dsn3 = self.crop(x, conv3_3.size())
        x = -1*(torch.sigmoid(crop1_dsn3))+1
        x = x.expand(-1, 256, -1, -1).mul(conv3_3)
        x = self.conv1_dsn3(x)
        x = F.relu(self.conv2_dsn3(x))
        x = F.relu(self.conv3_dsn3(x))
        x = self.conv4_dsn3(x) + crop1_dsn3
        upscore_dsn3 = self.crop(self.sum_dsn3_up(x), x_size)

        x = self.sum_dsn3_2(x)
        crop1_dsn2 = self.crop(x, conv2_2.size())
        x = -1*(torch.sigmoid(crop1_dsn2))+1
        x = x.expand(-1, 128, -1, -1).mul(conv2_2)
        x = self.conv1_dsn2(x)
        x = F.relu(self.conv2_dsn3(x))
        x = F.relu(self.conv2_dsn3(x))
        x = self.conv4_dsn2(x) + crop1_dsn2
        upscore_dsn2 = self.crop(self.sum_dsn2_up(x), x_size)

        x = -1*(torch.sigmoid(upscore_dsn2))+1
        x = x.expand(-1, 64, -1, -1).mul(conv1_2)
        x = self.conv1_dsn1(x)
        x = F.relu(self.conv2_dsn1(x))
        x = F.relu(self.conv3_dsn1(x))
        x = self.conv4_dsn1(x) + upscore_dsn2
        upscore_dsn1 = self.crop(self.sum_dsn2_up(x), x_size)

        return torch.sigmoid(upscore_dsn1), torch.sigmoid(upscore_dsn2), torch.sigmoid(upscore_dsn3), torch.sigmoid(upscore_dsn4), torch.sigmoid(upscore_dsn5), torch.sigmoid(upscore_dsn6)

    def train(self, batch_x, batch_y):
        dsn1, dsn2, dsn3, dsn4, dsn5, dsn6 = self.forward(batch_x)
        loss = nn.MSELoss()
        loss_1 = loss(dsn1, batch_y)
        loss_2 = loss(dsn2, batch_y)
        loss_3 = loss(dsn3, batch_y)
        loss_4 = loss(dsn4, batch_y)
        loss_5 = loss(dsn5, batch_y)
        loss_6 = loss(dsn6, batch_y)
        total_loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6
        self.optim.zero_grad()
        total_loss.backward()
        self.optim.step()
        return total_loss

    def test(self, batch_x):
        dsn1, dsn2, dsn3, dsn4, dsn5, dsn6 = self.forward(batch_x)
        dsn = dsn1.detach()+dsn2.detach()+dsn3.detach()+dsn4.detach()+dsn5.detach()+dsn6.detach()
        return dsn/6

    def crop(self, upsampled, x_size):
        c = (upsampled.size()[2] - x_size[2]) // 2
        _c = x_size[2] - upsampled.size()[2] + c
        assert(c >= 0)
        return upsampled[:, :, c:_c, c:_c]

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
