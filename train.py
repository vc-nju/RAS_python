import cv2
import numpy as np
import torch
import torch.utils.data as Data

from model import RAS
from utils import trans_im, trans_gt

IMAGES_NUM = 3372
MINI_IMAGES_NUM = 1120
BATCH_SIZE = 1
EPOCHS_NUM = 200


def get_train_data(start_image_id, end_image_id):
    image_num = end_image_id - start_image_id
    x = np.zeros([image_num, 3, 500, 500])
    y = np.zeros([image_num, 1, 500, 500])
    for i in range(image_num):
        im_path = "data/train/{}.jpg".format(i+start_image_id)
        gt_path = "data/train/{}.png".format(i+start_image_id)
        x[i, :, :, :] = trans_im(im_path)
        y[i, :, :, :] = trans_gt(gt_path)
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    return loader


if __name__ == "__main__":
    ras = RAS()
    ras.cuda()

    for epoch in range(99, EPOCHS_NUM):
        loader_num = IMAGES_NUM//MINI_IMAGES_NUM
        for l in range(loader_num):
            start_image_id = MINI_IMAGES_NUM * l
            end_image_id = MINI_IMAGES_NUM * (l+1)
            loader = get_train_data(start_image_id, end_image_id)
            for step, (batch_x, batch_y) in enumerate(loader):
                loss = ras.train(batch_x.cuda(), batch_y.cuda())
                _step = step + start_image_id
                if _step % 20 == 0:
                    print("epoch = {}, step = {}, loss = {}".format(
                        epoch, _step, loss))
                if _step % 100 == 0:
                    torch.save(ras.state_dict(), 'data/model/params.pkl')
            del(loader)
        torch.save(ras.state_dict(),
                   'data/model/epoch_{}_params.pkl'.format(epoch))
