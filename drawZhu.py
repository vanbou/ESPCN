# import matplotlib.pyplot as plt
#
# # 这两行代码解决 plt 中文显示的问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# keywords = ('model1', 'model2', 'model3', 'model4')
# params = [12809, 11491, 11925, 13245]
# flops = [415.06, 348.92, 353.84, 420.07]
#
# plt.bar(keywords, params, width=0.35)
# # plt.bar(keywords, params, width=0.2)
# for i in range(4):
#   plt.text(i,params[i],"%s"%params[i],va='center')
# # plt.bar(keywords, params)
# plt.title('参数对比')
#
# plt.show()

import torch

x = torch.randn(1,3,2,2)
y = torch.tensor([[[[1.0]],

         [[2.0]],

         [[3.0]]]]) #1,3,1,1
# print(y.shape)
# z = torch.mul(x,y)
# print(x,'\n',y,'\n',z)
# print(z.shape)#1,3,2,2
# print()
# a = x*y
# print(a.shape)
# print(a)
a = x+y
b = torch.add(x,y)
print(a)
print(b)