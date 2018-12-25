import cv2
import numpy as np
import torch
import torch.utils.data as Data

from model import RAS

IMAGES_NUM = 3372
MINI_IMAGES_NUM = 300
BATCH_SIZE = 12
EPOCHS_NUM = 200
LR_RATE = 0.0001

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

def get_train_data(start_image_id, end_image_id):
    image_num = end_image_id - start_image_id
    x = np.zeros([image_num, 3, 500, 500])
    y = np.zeros([image_num, 3, 500, 500])
    for i in range(image_num):
        im_path = "data/train/{}.jpg".format(i+start_image_id)
        gt_path = "data/train/{}.png".format(i+start_image_id)
        x[i,:,:,:] = trans_im(im_path)
        y[i,:,:,:] = trans_gt(gt_path)
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
    
    for epoch in range(EPOCHS_NUM):
        loader_num = IMAGES_NUM//MINI_IMAGES_NUM
        for l in range(loader_num):
            start_image_id = MINI_IMAGES_NUM * l
            end_image_id = MINI_IMAGES_NUM * (l+1)
            loader = get_train_data(start_image_id, end_image_id)
            for step, (batch_x, batch_y) in enumerate(loader):
                loss = ras.train(batch_x.cuda(), batch_y.cuda())
                if step%20 == 0:
                    print("epoch = {}, step = {}, loss = {}".format(epoch, step, loss))
            del(loader)
        torch.save(ras.state_dict(), 'data/model/params.pkl')