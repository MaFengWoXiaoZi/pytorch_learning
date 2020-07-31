# -*- coding: utf-8 -*-
'''
transforms使用
'''
#%%

import os 
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from my_dataset import RMBDataset
from PIL import Image
from matplotlib import pyplot as plt

os.chdir('E:\pytorch_learning')

def set_seed(seed = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
# 设置随机数 种子
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
    将data进行反transfrom操作
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
        raise Exception('Invalid img shape, expected 1 or 3 in axis 2, but got {0}' .format(img_.shape[2]))
    
    return img_


split_dir = os.path.join('第二章', 'data', 'rmb_split')
train_dir = os.path.join(split_dir, 'train')
valid_dir = os.path.join(split_dir, 'valid')

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    
    # 1. CenterCrop 从图像中心裁剪图片, size为裁剪图像尺寸
    # transforms.CenterCrop(512),
    
    # 2. RandomCrop 从图片中随机裁剪出尺寸为size的图片
    # padding是设置填充大小, 当为a时, 左右上下均为填充a个像素
    # 当为(a, b)时, 左右填充a个像素, 上下填充b个像素
    # 当为(a, b, c, d)时, 左上右下分别填充a,b,c,d个像素
    # pad_if_need 只有图像小于设定的size才会填充
    # padding_mode为填充模式, 四种模式: constant, edge, reflect, symmetric
    # constant像素值由fill决定, edge像素值由图像边缘决定
    # reflect表示镜像填充, 最后一个像素不镜像: eg:[1, 2, 3, 4] -> [3, 2, 1, 2, 3, 4, 3, 2]
    # symmetric表示镜像填充, 最后一个像素镜像: eg:[1, 2, 3, 4] -> [2, 1, 1, 2, 3, 4, 4, 3]
    # fill 设置填充的像素值, 需要padding_mode设置为constant, eg: (255, 0, 0)表示红色
    # transforms.RandomCrop(224, padding = 16),
    # transforms.RandomCrop(224, padding = (16, 64)),
    # transforms.RandomCrop(224,  padding=16, fill=(255, 0, 0), padding_mode = 'constant'),
    # transforms.RandomCrop(512, pad_if_needed=True),   # pad_if_needed=True
    # transforms.RandomCrop(224, padding=64, padding_mode='edge'),
    # transforms.RandomCrop(224, padding=64, padding_mode='reflect'),
    # transforms.RandomCrop(224, padding=64, padding_mode='symmetric'),
    
    # 3. RandomResizedCrop 随机大小、长宽比裁剪图片, size为裁剪尺寸
    # scale为随机裁剪面积比例, 默认为(0.08, 1)
    # ratio为随机长宽比, 默认(3/4, 4/3)
    # interpolation插值方法
    # PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC
    # transforms.RandomResizedCrop(size = 224, scale = (0.5, 0.5)),
    
    # 4. FiveCrop, TenCrop在图像上下左右以及中心裁剪出尺寸为size的5张图片
    # TenCrop对这5张图片进行水平或者垂直镜像获得10张图片
    # size为裁剪尺寸, vertical_flip为是否垂直翻转
    # transforms.FiveCrop(112),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),
    # transforms.TenCrop(112, vertical_flip = False),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),
    
    # 5. RandomHorizontalFlip和RandomVerticalFlip 依据概率p进行水平、垂直翻转
    # transforms.RandomHorizontalFlip(p = 1),
    # transforms.RandomVerticalFlip(p = 1),
    
    # 6. RandomRotation随机旋转图片, degrees为旋转角度
    # resample为重采样方法, expand为是否扩大图片
    # transforms.RandomRotation(90),
    # transforms.RandomRotation((90), expand = True),
    # transforms.RandomRotation(30, center = (0, 0)),
    #transforms.RandomRotation(30, center = (0, 0), expand = True),
    
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 构建MyDataSet实例
train_data = RMBDataset(data_dir = train_dir, transform = train_transform)
valid_data = RMBDataset(data_dir = valid_dir, transform = valid_transform)

# 构建DataLoader
train_loader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)
valid_loader = DataLoader(dataset = valid_data, batch_size = BATCH_SIZE)

# 训练模型
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        img_tensor = inputs[0, ...]
        img = transform_invert(img_tensor, train_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()        