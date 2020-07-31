# -*- coding: utf-8 -*-
'''
transforms使用2
'''

import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import random
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from my_dataset import RMBDataset
from PIL import Image

os.chdir('E:\\pytorch_learning')

def set_seed(seed = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
# 设置随机种子
set_seed(1)

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {'1': 0, '100': 1}

def transform_invert(img_, transform_train):
    '''
    将data进行反transform操作
    '''
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype = img_.dtype, device = img_.device)
        std = torch.tensor(norm_transform[0].std, dtype = img_.dtype, device = img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])
        
    img_ = img_.transpose(0, 2).transpose(0, 1)
    img_ = np.array(img_) * 255
        
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception('Invalid img shape, expected 1 or 3 in axis 2, but got {0}!' .format(img_.shape[2]))

    return img_
    
# 1. 构建数据
split_dir = os.path.join('第二章', 'data', 'rmb_split')
train_dir = os.path.join(split_dir, 'train')
valid_dir = os.path.join(split_dir, 'valid')

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    
    # 1. Pad对图像边缘进行填充
    # padding为设置填充大小, 当为a表示左右上下填充a个像素
    # 当为(a, b)时, 左右填充a个像素, 上下填充b个像素
    # 当为(a, b, c, d)时, 左上右下分别填充a, b, c, d个像素
    # padding_mode为填充模式, 有constant, edge, reflect, symmetric四种
    # fill: 当padding_mode为constant时, 设置填充的像素值为(R, G, B)或(Gray)
    # transforms.Pad(padding = 32, fill = (255, 0, 0), padding_mode = 'constant'),
    # transforms.Pad(padding = (8, 64), fill = (255, 0, 0), padding_mode = 'constant'), 
    # transforms.Pad(padding = (8, 16, 32, 64), fill = (255, 0, 0), padding_mode = 'constant'),
    # transforms.Pad(padding = (8, 16, 32, 64), fill = (255, 0, 0), padding_mode = 'symmetric'),
    # transforms.Pad(padding = (8, 16, 32, 64), fill = (255, 0, 0), padding_mode = 'reflect'),
    
    # 3. ColorJitter用于调整亮度, 饱和度和色相
    # brightness为亮度调整因子, 当为a时, 从[max(0, 1-a), 1+a]中随机选择
    # 当为(a, b)时, 表示[a, b]
    # contrast为对比度参数, saturation为饱和度参数
    # hue为色相参数, 当为a时, 从[-a, a]中选择参数 0<=a<=0.5
    # transforms.ColorJitter(brightness = 0.5),
    # transforms.ColorJitter(contrast = 0.5),
    # transforms.ColorJitter(saturation = 2),
    # transforms.ColorJitter(hue = 0.3),
    
    # 4. Grayscale依概率将图片转换为灰度图
    # num_output_channels, 表示输出通道数, 只能设为1或者3
    # p表示图像被转换为灰度的概率值
    # transforms.Grayscale(num_output_channels = 3),
    
    # 5. RandomAffine对图像进行仿射变换, 包括旋转, 平移, 缩放, 错切, 翻转
    # degrees表示旋转角度, translate表示平移区间设置, eg: (a, b), 其中a为宽, b为高
    # 图像在宽维度平移的区间为-img_width * a < dx < img_width * a
    # scale表示缩放比例, 以面积为单位, fillcolor表示填充的颜色
    # transforms.RandomAffine(degrees = 30),
    # transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2), fillcolor = (255, 0, 0)),
    # transforms.RandomAffine(degrees = 0, scale = (0.7, 0.7)),
    # transforms.RandomAffine(degrees = 0, shear = (0, 0, 0, 45)),
    # transforms.RandomAffine(degrees = 0, shear = 90, fillcolor = (255, 0, 0)),
    
    # 6. RandomErasing对图像进行随机遮挡
    # p为执行的概率值, scale为遮挡区域的面积比例, ratio为遮挡区域的长宽比, value设置遮挡区域的填充颜色
    # transforms.ToTensor(),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='1234'),
    
    # 7. RandomChoice从一组transforms操作中选择一个
    # transforms.RandomChoice([transforms.RandomVerticalFlip(p = 1), transforms.RandomHorizontalFlip(p = 1)]),
    
    # 8. RandomApply依据概率执行一组transforms操作
    transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=45, fillcolor=(255, 0, 0)), transforms.Grayscale(num_output_channels=3)], p=0.5), 
    
    # 9. RandomOrder对一组Transform操作打乱顺序
    transforms.RandomOrder([transforms.RandomRotation(15), transforms.Pad(padding=32), transforms.RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(0.9, 1.1))]), 
    
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])


# 2. 构建MyDataset实例
train_data = RMBDataset(data_dir = train_dir, transform = train_transform)
valid_data = RMBDataset(data_dir = valid_dir, transform = valid_transform)

# 3. 构建DataLoader
train_loader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)
valid_loader = DataLoader(dataset = valid_data, batch_size = BATCH_SIZE)

# 4. 训练模型
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        img_tensor = inputs[0, ...]
        img = transform_invert(img_tensor, train_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    