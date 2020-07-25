# -*- coding: utf-8 -*-
"""
自动求导
"""

import torch
torch.manual_seed(10)

flag = False

# 1. retain_graph保存计算图, 便于进行二次反向传播
if flag:
    w = torch.tensor([1.], requires_grad = True)
    x = torch.tensor([2.], requires_grad = True)
    
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)
    
    y.backward(retain_graph = True)
    print(w.grad)
    y.backward()

# 2. 反向传播时, 如果调用backward方法的是一个标量, 则不需要指定gradient参数, 如果调用backward方法的是一个向量, 则需要指定gradient参数
flag = False
if flag:
    w = torch.tensor([1.], requires_grad = True)
    x = torch.tensor([2.], requires_grad = True)
    
    a = torch.add(w, x)
    b = torch.add(w, 1)
    
    y0 = torch.mul(a, b) # y0 = (x + w) * (w + 1)
    y1 = torch.add(a, b) # y1 = (x + w) + (w + 1)  dy1/dw = 2
    
    loss = torch.cat([y0, y1], dim = 0)
    print(loss.size())
    grad_tensors = torch.tensor([1., 2.])
    
    loss.backward(gradient = grad_tensors) # gradient传入torch.autograd.backward()中的grad_tensors
    print(w.grad)
    print(x.grad)

# 3. 自动求导autograd.grad
flag = False
if flag:
    x = torch.tensor([3.], requires_grad = True)
    y = torch.pow(x, 2) # y = x ** 2
    grad_1 = torch.autograd.grad(y, x, create_graph = True)
    print(grad_1)
    grad_2 = torch.autograd.grad(grad_1[0], x)
    print(grad_2)
    
# 4. 反向传播中梯度会累加, 每次计算时需要将梯度清零
flag = False
if flag:
    w = torch.tensor([1.], requires_grad = True)
    x = torch.tensor([2.], requires_grad = True)
    
    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)
        
        y.backward()
        print(w.grad)
        
        # 对梯度清零
        w.grad.zero_()
        
# 5. requires_grad表示是否需要计算梯度, 依赖于叶子结点的结点默认该属性为True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad = True)
    x = torch.tensor([2.], requires_grad = True)
    
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)
    
    print(a.requires_grad, b.requires_grad, y.requires_grad)
    

# 6. +=操作与与原张量共享内存
flag = False
if flag:
    a = torch.ones((1, ))
    print(id(a), a)
    
    # 与原张量不共享内存
    a = a + torch.ones((1, ))
    print(id(a), a)
    # 与原张量共享内存
    a += torch.ones((1, ))
    print(id(a), a)
    
# 7. 叶子结点不可以执行in-place
flag = True
if flag:
    w = torch.tensor([1.], requires_grad = True)
    x = torch.tensor([2.], requires_grad = True)
    
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)
    
    a.add_(1)
    y.backward()
    
    
