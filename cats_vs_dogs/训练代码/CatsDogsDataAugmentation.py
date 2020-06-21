# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 22:57
# @Author  : Apokar
# @Email   : Apokar@163.com
# @File    : CatsDogsDataAugmentation.py
# @Comment :
from keras_preprocessing.image import ImageDataGenerator
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
datagen = ImageDataGenerator(
    rotation_range=40, # 角度值 0-180之间,表示图像随机旋转的角度范围
    width_shift_range=0.25, # 图像在 水平方向移动的范围，相对于总宽度
    height_shift_range=0.25, # 图像在 垂直方向移动的范围，相对于总高度
    shear_range=0.2, # 随机错切变换的角度
    zoom_range=0.2, # 图像随机缩放的范围
    horizontal_flip=True, # 随机将一半图像水平翻转
    fill_mode='nearest' # 填充新创建像素的方法，可能来源于旋转或宽度、高度平移
)

# 显示几个随机增强后的训练图像
import matplotlib.pyplot as plt

from keras.preprocessing import image

img_path = r"E:\Code\DL_chap5\cats_vs_dogs\数据\cats_and_dogs_small\train\cats\cat.5.jpg"  # 挑选一张图片
img = image.load_img(img_path, target_size=(150,150))  # 读取图像，并调整大小

x = image.img_to_array(img) # 将图片转换为形状为(150,150,3)的numpy数组

print("=======================================================")
x = x.reshape((1,)+x.shape) # 将图片转换为形状为(1,150,150,3)的numpy数组

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img((batch[0])))
    i += 1
    if i % 4 == 0:
        break

plt.show()
