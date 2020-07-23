# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import numpy as np

torch.manual_seed(1)

# 1. 通过torch创建张量
flag = True

if flag:
    arr = np.ones((3, 3))
    print("ndarry的数据类型: ", arr.dtype)
    t = torch.tensor(arr)
    print(t)
    
# 2. 从torch.from_numpy创建张量
if flag:
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    t = torch.from_numpy(arr)
    print("tensor: {0}" .format(t))
    # arr与t共享一块内存
    arr[0][0] = 100
    print("arr: {0}" .format(arr))
    print("arr: {0}" .format(t))
    
# 3. 通过torch.zeros创建张量
if flag:
    out_t = torch.tensor([[1]])
    t = torch.zeros((3, 3), out = out_t)
    print(t, '\n', out_t)
    print(id(t), id(out_t), id(t) == id(out_t))

# 4. 通过torch.full创建全1张量
if flag:
    t = torch.full((3, 3), 1)
    print(t)
    
# 5. 通过torch.arange创建等差数列张量, 不包含end
if flag:
    t = torch.arange(2, 10, 2)
    print(t)

# 6. 通过torch.linspace创建均分数列张量, 包含end  
if flag:
    t = torch.linspace(2, 10, 5)
    print(t)
    
# 7. 创建正态分布张量
if flag:
    # mean  std
    # 张量 张量
    mean = torch.arange(1, 5, dtype=torch.float)
    std = torch.arange(1, 5, dtype=torch.float)
    t_normal = torch.normal(mean, std)
    print('mean:{0}\nstd:{1}'.format(mean, std))
    print(t_normal)
    
    # 标量 标量
    t_normal = torch.normal(0., 1., size=(4, ))
    print(t_normal)    
    
    # 张量 标量
    mean = torch.arange(1, 5, dtype=torch.float)
    std = 1
    t_normal = torch.normal(mean, std)
    print('mean:{0}\nstd:{1}'.format(mean, std))
    print(t_normal)
    
    # 标量 张量
    mean = 0.
    std = torch.arange(1, 5, dtype=torch.float)
    t_normal = torch.normal(mean, std)
    print('mean:{0}\nstd:{1}'.format(mean, std))
    print(t_normal)