![](https://test-1253607195.cos.ap-shanghai.myqcloud.com/2019-1-1/logo.png)
# RAS-pytorch

Pytorch realization for "Reverse Attention for Salient Object Detection": [ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Shuhan_Chen_Reverse_Attention_for_ECCV_2018_paper.pdf).

# Preview
![Origin_photo](https://test-1253607195.cos.ap-shanghai.myqcloud.com/2019-1-1/1.jpg)
![Saliency_map](https://test-1253607195.cos.ap-shanghai.myqcloud.com/2019-1-1/1.png)

# Feature
RAS_pytorch is a pytorch version for the paper mentioned above.

We have trained and tested on MSRA-B, and it's auc is 0.976.

# Requirements

- python 3.5+
- opencv 3.0+
- pytorch 0.4+

# Installation

```bash
git clone https://github.com/vc-nju/RAS_python.git && cd RAS_python
mkdir data && mkdir data/model && mkdir data/visualization
```
The pre_train models can be downloaded from [Google Drive]() and [BaiduYun](). Please copy them to data/model/

# Test Zoo

Let's take a look at a quick example.

0. Make sure you have downloaded the models and copy them to data/model/

Your data/model should be like this:
```
drfi_python
└───data
    └───model
        |  epoch_99_params.pkl
```

1. Edit ./test.py module in your project:

```python
    # img_path and id can be replaced by yourself.
    TEST_ID = 914
    ...
    im_path = "data/test/{}.jpg".format(TEST_ID)
    gt_path = "data/test/{}.png".format(TEST_ID)
```

2. Running test using python3:
```bash
python3 test.py
```

# Training

1. Edit ./train.py in your project:

```python
def get_train_data(start_image_id, end_image_id):
    """
    add your load_data code here.
    """
```
2. Running train using python3:

```bash
python3 train.py
```

# Validation

1. Edit ./val.py in your project:

```python
def get_val_data(start_image_id, end_image_id):
    """
    add your load_data code here.
    """
```
2. Running validation using python3:

```bash
python3 val.py
```