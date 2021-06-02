import matplotlib.pyplot as plt
import numpy as np
import time
# with open('loss.txt', 'w') as f:
#     for i in range(5):
#         f.write(str(i))
#         time.sleep(1)
#         print(i)
#     f.close()


import numpy as np
from tensorboardX import SummaryWriter




f = open(r"5.txt")
line = f.readline()
a = []
b = []
while line:
    num1, num2 = list(map(float,line.split(" ")))
    a.append(num1)
    b.append(num2)
    line = f.readline()
f.close()
a = np.array(a)
a = a.astype(int)
b = np.array(b)

writer = SummaryWriter()
loss = np.random.randn(10)
# for i, val in enumerate(b):
#     writer.add_scalar(tag='Checking range', scalar_value=val, global_step=i)
from FSRCNN.model import Net
models = Net(num_channels=1,upscale_factor=4)

import torch
dummy_input = torch.rand(1, 1, 64,64)
with SummaryWriter(comment='fsrcnn') as w:
    w.add_graph(models, (dummy_input,))
writer.close()

#
# plt.plot(a, b, 'r-')
# plt.tight_layout
# plt.title('FSRCNN PSNR')
# plt.xlabel('Epochs')
# plt.ylabel('PSNR')
# plt.legend(["origin"])
#
# y1_max=np.argmax(b)
# va = round(b[y1_max], 2)
# show_max='['+str(va)+']'
# # 以●绘制最大值点和最小值点的位置
# plt.plot(y1_max,va,'ko')
# plt.annotate(str(va), xy=(y1_max,va), xytext=(y1_max,va))
# plt.savefig('psnr.png',dpi=1000)
# plt.show()



# with open(file, 'r') as f:
#     data = f.readlines()  # 将txt中所有字符串读入data
#     for line in data:
#         numbers_float = map(float, line)  # 转化为浮点数
#         loss.append(numbers_float)
# print(list(loss))
# np.array(list(map(float)), dtype=np.float32)

# with open("loss.txt", "r") as f:  # 打开文件
#     data = f.read()  # 读取文件
#
# for i in range(0,100):
#
# with open("psnr.txt", "r") as f1:
#     psnr = f1.read()
#     map(float,psnr)
#
# with open("test.txt", "w") as f:
#     for i in range(0, 200):
#         n = random.randint(30, 80)
#         m = random.randint(30, 80)
#         Loss_list.append(n)
#         Accuracy_list.append(m)
#         f.write("" + str(n))
#         f.writelines("\n")
#     # Accuracy_list[i] = random.random()
#
# x1 = []
# x2 = []
# for i in range(1,101):
#     x1.append(i)
#     x2.append(i)
# # print(x1)
# # print(x2)
# y1 = loss
# y2 = psnr
#
# plt.plot(x1, y1)

# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'y-')
# plt.tight_layout
# plt.title('FSRCNN PSNR&LOSS')
# plt.xlabel('Epochs')
# plt.ylabel('PSNR')
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, 'b-')
# plt.tight_layout
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()
# plt.savefig("accuracy_loss.jpg")
