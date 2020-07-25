# -*- coding: utf-8 -*-
"""
计算图与梯度求导
"""

import torch

w = torch.tensor([1.], requires_grad = True)
x = torch.tensor([2.], requires_grad= True)

a = torch.add(w, x)
# 保存中间节点梯度
a.retain_grad()
b = torch.add(w, 1)
b.retain_grad()
y = torch.mul(a, b)
y.retain_grad()

y.backward()
print(w.grad)

# 查看叶子节点
print('is_leaf: \n', w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# 查看梯度
print('gradient: \n', w.grad, x.grad, a.grad, b.grad, y.grad)

# 查看创建该张量时所用的方法
print('grad_fn: \n', w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)