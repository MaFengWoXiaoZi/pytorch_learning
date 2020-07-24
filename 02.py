# -*- coding: utf-8 -*-
"""
张量操作
"""

import torch
from matplotlib import pyplot as plt
import numpy as np

# 1. 将张量按照维度dim进行拼接, arange方法从start到end-1, 不产生新的维度
flag = False

if flag:
    tensor = torch.arange(1, 10).reshape(3, 3)
    print(tensor)
    res = torch.cat([tensor, tensor], dim = 0)
    print(res, res.size())
    res = torch.cat([tensor, tensor], dim = 1)
    print(res, res.size)

# 2. 在新创建的维度dim上进行拼接, 以两个3×3矩阵为例, 若在第0维度拼接, cat方法产生6*3的矩阵, 而stack则先产生1×3×3矩阵, 然后拼接成为2×3×3矩阵
if flag:
    tensor = torch.arange(1, 10).reshape(3, 3)
    res = torch.stack([tensor, tensor], dim = 0)
    print(res, res.size())
    res = torch.stack([tensor, tensor], dim = 1)
    print(res, res.size())
    res = torch.stack([tensor, tensor], dim = 2)
    print(res, res.size())

# 3. 将张量按照维度dim进行平均切分, 若不能整除, 最后一份张量小于其他张量
if flag:
    tensor = torch.arange(1, 10).reshape(3, 3)
    print(torch.chunk(tensor, 3, dim = 0))
    res = torch.chunk(tensor, 3, dim = 1)
    print(res[0], res[0].size())

    tensor2 = torch.randint(0, 10, (2, 5))
    print("tensor2: ", tensor2)
    print(torch.chunk(tensor2, 2, dim = 1))
    print(torch.chunk(tensor2, 2, dim = 0))
    print(torch.chunk(tensor2, 4, dim = 1))

# 将张量按照维度dim进行切分, 并且给出一个切分每一块长度的列表
if flag:
    res = torch.split(tensor, [2, 1], dim = 0)
    print(res)
    res = torch.split(tensor, [1, 2], dim = 1)
    print(res)

# 在维度dim上, 按照index索引数据, 这里index也需要是一个tensor张量
if flag:
    tensor = torch.randint(0, 9, size = (3, 3))
    print(tensor)
    idx = torch.tensor([0, 2], dtype = torch.long)
    print(idx)
    print(torch.index_select(tensor, dim = 0, index = idx))

    tensor = torch.randint(10, 100, size = (5, 2))
    print(tensor)
    print(torch.index_select(tensor, dim = 0, index = torch.tensor([0, 1], dtype = torch.int64)))
    print(torch.index_select(tensor, dim = 1, index = torch.tensor([1])))

# 按照mask中的True进行索引, mask产生一个同输入张量input相同的布尔类型张量
if flag:  
    tensor = torch.randint(0, 9, size = (3, 3))
    print(tensor)
    mask = tensor.ge(5)
    print(mask)
    print(torch.masked_select(tensor, mask))

# 变换张量形状, 新张量与原张量共享内存空间
if flag:
    # randperm产生从0到n-1的随机排列, 1*n
    tensor = torch.randperm(8)
    t4_reshape = tensor.reshape(shape = (4, 2))
    t4_reshape[0][0] = 100
    print(t4_reshape)
    print(tensor)
    res = torch.reshape(tensor, (-1, 2, 2))
    res[0][0][0] = 9999
    print(res)
    print(tensor)
    print(t4_reshape)
    print(id(tensor), id(t4_reshape), id(res))

# 交换张量的两个维度
if flag:
    tensor = torch.randint(0, 30, size = (5, 2, 3))
    print(tensor, tensor.size())
    print(torch.transpose(tensor, dim0 = 1, dim1 = 2))
    print(torch.transpose(tensor, dim0 = 0, dim1 = 1))
    # t方法用于对矩阵转置
    matrix = torch.randperm(8).reshape(4, 2)
    print(matrix)
    print(torch.t(matrix))
    print(matrix.t())

# 压缩长度为1的维度
if flag:
    tensor = torch.rand((1, 2, 3, 1))
    print(tensor)
    print(torch.squeeze(tensor).size())
    print(torch.squeeze(tensor, dim = 0).size())
    print(torch.squeeze(tensor, dim = 1).size())

# 依据dim来扩展维度
if flag:
    tensor = torch.rand((2, 2)) * 10
    print(tensor)
    print(torch.unsqueeze(tensor, dim = 0).size())
    print(tensor.unsqueeze(dim = 1).size())

# 张量的数学运算 加法
if flag:
    t1 = torch.round(torch.rand(2, 2) * 10)
    t2 = torch.round(torch.rand(2, 2) * 10)
    print(t1, '\n', t2)
    print(t1.add(t2))
    print(t1.add(alpha = 2, other = t2))
    print(torch.add(input = t1, alpha = 2, other = t2))

    print('t1: ', t1, '\n', 't2: ', t2)
    t3 = torch.round(torch.rand(2, 2) * 10)
    print('t3: ', t3)
    print(torch.addcmul(t3, t1, t2, value = 2))
    print(torch.addcdiv(t3, t1, t2, value = 1))
 
# 张量的数学运算 减法、乘法(点乘)、除法
if flag:
    t1 = torch.randint(0, 100, (2, 2))
    t2 = torch.randint(0, 100, (2, 2))
    print(t1, t2)
    print(torch.sub(t1, t2))
    
    t3 = torch.tensor([[1, 2], [3, 4]])
    t4 = torch.tensor([[1, 2], [3, 4]])
    print(t3.mul(t4))
    print(t3.div(t4))

flag = True    
# 张量的数学运算 对数、指数、三角函数
if flag:
    t = torch.tensor([np.e, np.e])
    print(torch.log(t))
    t = torch.tensor([[10., 1.], [100., 10.]])
    print(torch.log10(t))
    t = torch.tensor([2., 1.])
    print(torch.log2(t))
    
    print(torch.exp(t))
    print(torch.pow(t, 3))
    
    t = torch.tensor([[-1, 2], [0, -9]])
    print(torch.abs(t))
    t = torch.tensor([np.pi, 0])
    print(torch.sin(t))
    print(torch.asin(t))
    print(torch.cos(t))
    print(torch.acos(t))
    print(torch.tan(t))
    print(torch.atan(t))