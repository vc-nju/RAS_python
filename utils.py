import cv2
import numpy as np


def trans_im(path):
    im = cv2.imread(path).astype(np.float)/255.
    im = cv2.resize(im, (500, 500), interpolation=cv2.INTER_AREA)
    x = np.zeros([1, 3, 500, 500])
    x[0, 0, :, :] = im[:, :, 0]
    x[0, 1, :, :] = im[:, :, 1]
    x[0, 2, :, :] = im[:, :, 2]
    return x


def trans_gt(path):
    im = cv2.imread(path).astype(np.float)/255.
    im = cv2.resize(im, (500, 500), interpolation=cv2.INTER_AREA)
    x = np.zeros([1, 1, 500, 500])
    x[0, 0, :, :] = im[:, :, 0]
    return x
