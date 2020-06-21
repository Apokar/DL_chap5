# -*- coding: utf-8 -*-
# @Time    : 2020/6/17 0:13
# @Author  : Apokar
# @Email   : Apokar@163.com
# @File    : CatsDogsModelVGG16.py
# @Comment : 将 VGG16 卷积基实例化

'''
在你的数据集上运行卷积基，将输出保存成硬盘中的 Numpy 数组，然后用这个数据作
为输入，输入到独立的密集连接分类器中（与本书第一部分介绍的分类器类似）。这种
方法速度快，计算代价低，因为对于每个输入图像只需运行一次卷积基，而卷积基是目
前流程中计算代价最高的。但出于同样的原因，这种方法不允许你使用数据增强。

'''

from keras.applications import VGG16

# 将VGG16卷积实例化
conv_base = VGG16(weights='imagenet',  # 指定模型初始化的权重检查点
                  include_top=False,  # 制定模型最后是否包含密集连接分类器，默认情况下，这个密集连接分类器对应ImageNet的1000个类别。而我们只有cat和dog，所以不需要包含它。
                  input_shape=(150, 150, 3)  # 不传入输入形状参数，则可以处理任意形状的输入
                  )

# conv_base.summary()

# 不使用数据增强的快速

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = r"E:\Code\DL_chap5\cats_vs_dogs\数据\cats_and_dogs_small"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))  # conv_base.summary() 可知特征图形状为 (4, 4, 512)
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break  # 生成器不断生成数据，所以读取所有图片后终止循环
    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
test_features, test_labels = extract_features(test_dir, 1000)
validation_features, validation_labels = extract_features(validation_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))

'''
保存你的数据在 conv_base 中的
输出，然后将这些输出作为输入用于新模型
'''


# 定义并训练 密集连接分类器

from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc']
              )

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
