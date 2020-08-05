import torch
import torch.nn as nn
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    if 'ToTensor' in str(transform_train):
        img_ = img_.detach().numpy() * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )

    return img_


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
# 设置随机数种子
set_seed(3)

# 加载图片
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'lena.png')
img = Image.open(path_img).convert('RGB')

# 将图片转换成张量
img_transform = transforms.Compose([
    transforms.ToTensor()
])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim = 0)

# 卷积
flag = 0
if flag:
    conv_layer = nn.Conv2d(3, 1, 3)
    nn.init.xavier_normal_(conv_layer.weight.data)
    img_conv = conv_layer(img_tensor)
    
# 转置卷积
flag = 0
if flag:
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride = 2)
    nn.init.xavier_normal_(conv_layer.weight.data)
    img_conv = conv_layer(img_tensor)
    
# 可视化
if __name__ == "__main__":
    print('卷积前尺寸: {0}\n卷积后尺寸: {1}' .format(img_tensor.shape, img_conv.shape))
    img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)
    img_raw = transform_invert(img_tensor.squeeze(), img_transform)
    plt.subplot(121).imshow(img_raw)
    plt.subplot(122).imshow(img_conv, cmap = 'gray')
    plt.show() 

    