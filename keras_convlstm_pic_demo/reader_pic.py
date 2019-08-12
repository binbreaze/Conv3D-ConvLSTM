#!/usr/bin/python
# -*- coding:utf8 -*-

from PIL import Image
import numpy as np
 
# 三张输入图片
I1 = Image.open('./precip_0000.png')
I2 = Image.open('./PBLH_0000.png')
I3 = Image.open('./SFC_TMP_0000.png')
# 一张输出图片
I4 = Image.open('./PM25_AI_0000.png')


I1_array = np.array(I1)
I2_array = np.array(I2)
I3_array = np.array(I3)
# 模拟t-4,t-3,t-2,t-1时刻的图片输入数据
x1 = np.concatenate((I1_array, I2_array, I3_array), axis=2)
x2 = np.concatenate((I1_array, I2_array, I3_array), axis=2)
x3 = np.concatenate((I1_array, I2_array, I3_array), axis=2)
x4 = np.concatenate((I1_array, I2_array, I3_array), axis=2)
# 构造一个输入样本
x = np.array([x1,x2,x3,x4])
# 构造10个输入样本
X = np.array([x,x,x,x,x,x,x,x,x,x])
# 构造一个输出样本
y = np.array(I4)
# 构造十个输出样本
Y = np.array([y,y,y,y,y,y,y,y,y,y])    
# print(X.shape)
# # print(X)
# print(Y.shape)