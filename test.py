import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# class RAS(nn.Module):
#     def __init__(self):
#         super(RAS, self).__init__()
#         self.conv1_1 = nn.Sequential( nn.Conv2d(3, 64, kernel_size=3, padding=1), F.relu())
#     def forward(self, x):
#         self.conv1_1(x)

def crop(upsampled, data):
    c = (upsampled.shape[2] - data.shape[2]) // 2
    return upsampled[:, :, c:-c, c:-c]

if __name__ == "__main__":
    # ras = RAS()
    # x = cv2.imread("13.jpg")
    # print(ras.forward())
    a = np.zeros([3,1,600,600])
    b = np.zeros([3,1,500,500])
    c = crop(a, b)
    print(c.shape)