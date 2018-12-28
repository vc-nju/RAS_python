import cv2
import numpy as np
import torch
import torch.utils.data as Data
from sklearn import metrics

from model import RAS
from utils import trans_im, trans_gt

IMAGES_NUM = 480
BATCH_SIZE = 1


def get_val_data(start_image_id, end_image_id):
    image_num = end_image_id - start_image_id
    x = np.zeros([image_num, 3, 500, 500])
    y = np.zeros([image_num, 1, 500, 500])
    im_ids = np.zeros([image_num, 1])
    for i in range(image_num):
        im_path = "data/val/{}.jpg".format(i+start_image_id)
        gt_path = "data/val/{}.png".format(i+start_image_id)
        x[i, :, :, :] = trans_im(im_path)
        y[i, :, :, :] = trans_gt(gt_path)
        im_ids[i] = i+start_image_id
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    im_ids = torch.Tensor(im_ids)
    torch_dataset = Data.TensorDataset(x, y, im_ids)
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
    ras.load_state_dict(torch.load("data/model/epoch_99_params.pkl"))

    Y_test = []
    Y_prob = []
    loader = get_val_data(0, IMAGES_NUM)
    for step, (batch_x, batch_y, im_id) in enumerate(loader):
        im_path_pre = "data/visualization/{}".format(int(im_id.numpy()[0, 0]))
        batch_y_prob = ras.test(batch_x.cuda(), im_path_pre)
        Y_test.append(batch_y.numpy().flatten().astype(np.int32))
        Y_prob.append(batch_y_prob.cpu().numpy().flatten())
        im_path = "data/visualization/{}.png".format(int(im_id.numpy()[0, 0]))
        im = Y_prob[-1].reshape(500, 500, 1)*255
        cv2.imwrite(im_path, im.astype(np.uint8))
        if step % 20 == 0:
            print("finished step {}".format(step))
    auc = metrics.roc_auc_score(
        np.array(Y_test).flatten(), np.array(Y_prob).flatten())
    print("auc is {}".format(auc))
