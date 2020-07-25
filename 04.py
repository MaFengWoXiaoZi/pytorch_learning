# -*- coding: utf-8 -*-
"""
线性回归模型2
"""
import torch
from matplotlib import pyplot as plt

torch.manual_seed(10)
lr = 0.01
best_loss = float('inf')

# 创建训练数据
x = torch.rand(200, 1) * 10
y = 3 * x + (5 + torch.randn(200, 1))

# 构建线性回归参数
w = torch.randn((1), requires_grad = True)
b = torch.zeros((1), requires_grad = True)

for iteration in range(10000):
    
    # 前向传播
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)
    
    # 计算 MSE loss
    loss = (0.5 * (y - y_pred) ** 2).mean()
    
    # 反向传播
    loss.backward()
    
    current_loss = loss.item()
    if current_loss < best_loss:
        best_loss = current_loss
        best_w = w
        best_b = b
        
        # 绘图
        if loss.data.numpy() < 3:
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
            plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict = {'size': 20, 'color': 'red'})
            plt.xlim(1.5, 10)
            plt.ylim(8, 40)
            plt.title('Iteration: {0}\nw:{1} b:{2}' .format(iteration, w.data.numpy(), b.data.numpy()))
            plt.pause(0.5)
            
            if loss.data.numpy() < 0.55:
                break
        
    # 更新参数
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)