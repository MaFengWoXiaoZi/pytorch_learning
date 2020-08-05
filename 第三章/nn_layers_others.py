# -*- coding: utf-8 -*-

'''
其他网络层: 池化层(最大池化、平均池化), 最大上采样, 线性变换
'''
import os
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
from nn_layers_convolution import transform_invert, set_seed

# 设置随机种子
set_seed(1)  

# 加载图片
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'lena.png')
img = Image.open(path_img).convert('RGB') 

# 转换为张量
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0) 

# 创建卷积层
# 最大池化 maxpool
flag = 1
if flag:
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2))   # input:(i, o, size) weights:(o, i , h, w)
    img_pool = maxpool_layer(img_tensor)

# 平均池化 avgpool
flag = 0
if flag:
    avgpoollayer = nn.AvgPool2d((2, 2), stride=(2, 2))   # input:(i, o, size) weights:(o, i , h, w)
    img_pool = avgpoollayer(img_tensor)

# 平均池化 avgpool 
# divisor_override 除法因子 eg: 若为3, 则为和除以3, 不指定则为和除以元素个数
flag = 0
if flag:
    img_tensor = torch.ones((1, 1, 4, 4))
    avgpool_layer = nn.AvgPool2d((2, 2), stride=(2, 2), divisor_override=3)
    img_pool = avgpool_layer(img_tensor)

    print("raw_img:\n{0}\npooling_img:\n{1}".format(img_tensor, img_pool))


# maxunpool 最大上采样
# maxpool中的return_indices以tensor形式返回所有最大值的下标
flag = 0
if flag:
    img_tensor = torch.randint(high=5, size=(1, 1, 4, 4), dtype=torch.float)
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
    img_pool, indices = maxpool_layer(img_tensor)

    img_reconstruct = torch.randn_like(img_pool, dtype=torch.float)
    maxunpool_layer = nn.MaxUnpool2d((2, 2), stride=(2, 2))
    img_unpool = maxunpool_layer(img_reconstruct, indices)

    print("raw_img:\n{0}\nimg_pool:\n{1}".format(img_tensor, img_pool))
    print("img_reconstruct:\n{0}\nimg_unpool:\n{1}".format(img_reconstruct, img_unpool))


# 线性变换 linear
flag = 0
if flag:
    inputs = torch.tensor([[1., 2, 3]])
    linear_layer = nn.Linear(3, 4)
    linear_layer.weight.data = torch.tensor([[1., 1., 1.],
                                             [2., 2., 2.],
                                             [3., 3., 3.],
                                             [4., 4., 4.]])

    linear_layer.bias.data.fill_(0.5)
    output = linear_layer(inputs)
    print(inputs, inputs.shape)
    print(linear_layer.weight.data, linear_layer.weight.data.shape)
    print(output, output.shape)


# 可视化
print("池化前尺寸:{0}\n池化后尺寸:{1}".format(img_tensor.shape, img_pool.shape))
img_pool = transform_invert(img_pool[0, 0:3, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_pool)
plt.subplot(121).imshow(img_raw)
plt.show()