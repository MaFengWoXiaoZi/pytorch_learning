
# -*- coding: utf-8 -*-
"""
线性回归模型
"""

import torch
from matplotlib import pyplot as plt

# 手动设置随机数种子
torch.manual_seed(10)

# 设置学习率
lr = 0.1

# rand产生均匀分布, randn产生标准正态分布, 创建20个数据点(x, y)
x = torch.rand(20 ,1) * 10
y = 2 * x + (5 + torch.randn(20, 1))

# 构建线性回归参数
w = torch.randn((1), requires_grad = True)
b = torch.zeros((1), requires_grad = True)

for iteration in range(1000):
    
    # 前向传播
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)
    
    # 计算均方差
    loss = (0.5 * (y - y_pred) ** 2).mean()
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)
    
    # 绘制图像
    if iteration % 20 == 0:
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw = 5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict = {'size': 20, 'color': 'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title('Iteration: {0}\n w: {1}, b: {2}' .format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)
        
        if loss.data.numpy() < 1:
            break
        