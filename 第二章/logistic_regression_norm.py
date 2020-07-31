# -*- coding: utf-8 -*-
'''
逻辑回归和标准化
'''

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
torch.manual_seed(10)

# 设置学习率
lr = 0.01

# 生成虚拟数据
sample_nums = 100
mean_value = 1.7
bias = 5
n_data = torch.ones(sample_nums, 2)
x0 = torch.normal(mean_value * n_data, 1) + bias
y0 = torch.zeros(sample_nums)
x1 = torch.normal(-mean_value * n_data, 1) + bias
y1 = torch.ones(sample_nums)
train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)

# 定义模型
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        
        return x
    
lr_net = LR()

# 定义损失函数与优化器
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(lr_net.parameters(), lr = 0.01, momentum = 0.9)

for iteration in range(1000):
    # 前向传播
    y_pred = lr_net(train_x)
    
    # 计算 MSE loss
    loss = loss_fn(y_pred, train_y)
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
    # 清空梯度
    optimizer.zero_grad()
    
    # 绘图
    if iteration % 40 == 0:
        # 以0.5为阈值进行分类
        mask = y_pred.ge(0.5).squeeze()
        # 计算正确预测的样本个数
        correct = (mask == train_y).sum()
        # 计算精度
        acc = correct.item() / train_y.size(0)
        
        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1])
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1])
        
        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1
        
        plt.xlim(-5, 10)
        plt.ylim(-7, 10)
        plt.plot(plot_x, plot_y)  
        
        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict = {'size': 20, 'color': 'red'})
        plt.title('Iteration: {0}\nw0:{1:.2f} w1:{2:.2f} b:{3:.2f} accuracy:{4:.2%}' .format(iteration, w0, w1, plot_b, acc))
        plt.legend()
        
        plt.show()
        plt.pause(0.5)
        
        if acc > 0.99:
            break     
