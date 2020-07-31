# -*- coding: utf-8 -*-
'''
    数据集的定义
'''

import os
import random
from PIL import Image
from torch.utils.data import Dataset

# 设置随机数种子
random.seed(1)
rmb_label = {'1': 0, '100': 1}

class RMBDataset(Dataset):
    def __init__(self, data_dir, transform = None, label_name = rmb_label):
        '''
        datadir表示数据集所在的目录
        transform表示对数据集使用预处理
        '''
        self.label_name = label_name
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
    
    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        
        if self.transform is not None:
            # 对图片进行变换
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return len(self.data_info)    
        
    def get_img_info(self, data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                # 获取sub_dir文件夹下的所有图片名, 保存到img_names列表中
                img_names = os.listdir(os.path.join(root, sub_dir))
                # 对获得的图片列表, 将图片绝对路径和图片的标签作为一个元祖添加到data_info列表中
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img, int(label)))
        
        return data_info
                
                
            
    