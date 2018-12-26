import cv2
import numpy as np
import torch
from sklearn import metrics

from model import RAS
from utils import trans_im

TEST_ID = 914

if __name__ == "__main__":
    ras = RAS()
    ras.cuda()
    ras.load_state_dict(torch.load("data/model/epoch_99_params.pkl"))
    
    im_path = "data/test/{}.jpg".format(TEST_ID)
    gt_path = "data/test/{}.png".format(TEST_ID)
    im_shape = cv2.imread(im_path).shape
    gt = cv2.imread(gt_path)[:, :, 0].astype(np.float)/255.
    x = trans_im(im_path)
    
    y_prob = ras.test(torch.FloatTensor(x).cuda())
    y_prob = y_prob.cpu().numpy()[0, 0, :, :]
    y_prob = cv2.resize(y_prob, (im_shape[1], im_shape[0]), interpolation=cv2.INTER_AREA)
    auc = metrics.roc_auc_score(gt.flatten(), y_prob.flatten())
    
    img = np.zeros([im_shape[0], im_shape[1]*3, 3])
    img[:, :im_shape[1], :] = cv2.imread(im_path).astype(np.float)/255.
    img[:, im_shape[1]:im_shape[1]*2, :] = gt.repeat(3).reshape([im_shape[0], im_shape[1], 3])
    img[:, im_shape[1]*2:, :] = y_prob.repeat(3).reshape([im_shape[0], im_shape[1], 3])
    
    print("finished~( •̀ ω •́ )y")
    print("auc is {}".format(auc))
    cv2.imshow("result", img)
    cv2.waitKey(0)
    

    
