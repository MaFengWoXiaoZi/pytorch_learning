# -*- coding: utf-8 -*-
'''
模型创建学习
''' 
import sys
import os
sys.path.append(os.getcwd())
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from 第二章.lenet import LeNet
from 第二章.my_dataset import RMBDataset
import numpy as np



def set_seed(seed = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
# 设置随机数种子
set_seed()
rmb_label = {'1': 0, '100': 1}

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 16
LR = 0.01
log_interval = 10
val_interval = 1

# 1. 获取数据
split_dir = os.path.join('第二章', 'data', 'rmb_split')
train_dir = os.path.join(split_dir, 'train')
valid_dir = os.path.join(split_dir, 'valid')

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding = 4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir = train_dir, transform = train_transform)
valid_data = RMBDataset(data_dir = valid_dir, transform = valid_transform)

# 构建DataLoader
train_loader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)
valid_loader = DataLoader(dataset = valid_data, batch_size = BATCH_SIZE)

# 2. 构建模型
net = LeNet(classes = 2)
net.initialize_weights()

# 3. 构建损失函数
criterion = nn.CrossEntropyLoss()

# 4. 构建优化器
optimizer = optim.SGD(net.parameters(), lr = LR, momentum = 0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)

# 5. 模型训练
train_curve = list()
valid_curve = list()

for epoch in range(MAX_EPOCH):
    loss_mean = 0.
    correct = 0.
    total = 0.
    
    net.train()
    for i, data in enumerate(train_loader):
        
        # forward
        inputs, labels = data
        outputs = net(inputs)
        
        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 更新权值
        optimizer.step()
        
        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()
        
        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i + 1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print('Training: Epoch[{0:0>3}/{1:0>3}] Iteration[{2:0>3}/{3:0>3}] Loss: {4:.4f} Acc:{5:.2%}' .format(
                epoch, MAX_EPOCH, i + 1, len(train_loader), loss_mean, correct / total))
            loss_mean = 0.
            
    # 更新学习率        
    scheduler.step()
    
    # 验证模型
    if (epoch + 1) % val_interval == 0:
        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().numpy()
                
                loss_val += loss.item()
                
            valid_curve.append(loss_val)
            print('Valid: \t Epoch[{0:0>3}/{1:0>3}] Iteration[{2:0>3}/{3:0>3}] Loss: {4:.4f} Acc:{5:.2%}' .format(
                epoch, MAX_EPOCH, j + 1, len(valid_loader), loss_val, correct / total))
            
        train_x = range(len(train_curve))
        train_y = train_curve
        
        train_iters = len(train_loader)
        # 因为valid中记录的是epochloss, 需要对记录点进行转换到iterations
        valid_x = np.arange(1, len(valid_curve) + 1) * train_iters * val_interval
        valid_y = valid_curve
        
plt.plot(train_x, train_y, label = 'Train')
plt.plot(valid_x, valid_y, label = 'Valid')

plt.legend(loc = 'upper right')
plt.xlabel('Iteration')
plt.ylabel('loss value')
plt.show()

BASE_DIR = os.path.join(os.path.absapth(__file__))
test_dir = os.path.join(BASE_DIR, 'test_data')

test_data = RMBDataset(data_dir = test_dir, transform = valid_transform)
valid_loader = DataLoader(dataset = test_data, batch_size = 1)

for i, data in enumerate(valid_loader):
    # forward
    inputs, labels = data
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    
    rmb = 1 if predicted.numpy()[0] == 0 else 100
    print('模型获得{0}元' .format(rmb))
        