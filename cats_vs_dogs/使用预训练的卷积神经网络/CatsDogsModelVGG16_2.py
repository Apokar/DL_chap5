# -*- coding: utf-8 -*-
# @Time    : 2020/6/21 16:53
# @Author  : Apokar
# @Email   : Apokar@163.com
# @File    : CatsDogsModelVGG16_2.py
# @Comment : 扩展 conv_base 模型，然后在输入数据上端到端地运行模型。

'''
在顶部添加 Dense 层来扩展已有模型（即 conv_base），并在输入数据上端到端地运行
整个模型。这样你可以使用数据增强，因为每个输入图像进入模型时都会经过卷积基。
但出于同样的原因，这种方法的计算代价比第一种要高很多。

'''

from keras.applications import VGG16

# 将VGG16卷积实例化
conv_base = VGG16(weights='imagenet',  # 指定模型初始化的权重检查点
                  include_top=False,  # 制定模型最后是否包含密集连接分类器，默认情况下，这个密集连接分类器对应ImageNet的1000个类别。而我们只有cat和dog，所以不需要包含它。
                  input_shape=(150, 150, 3)  # 不传入输入形状参数，则可以处理任意形状的输入
                  )

# conv_base.summary()

# 模型的行为和层类似，所以你可以向 Sequential 模型中添加一个模型（比如 conv_base），就像添加一个层一样。
# 在卷积基上添加一个密集连接分类器
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Model)                (None, 4, 4, 512)         14714688  
_________________________________________________________________
flatten_1 (Flatten)          (None, 8192)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               2097408   
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 257       
=================================================================
Total params: 16,812,353
Trainable params: 16,812,353
Non-trainable params: 0
_________________________________________________________________
如你所见， VGG16 的卷积基有 14 714 688 个参数，非常多。在其上添加的分类器有 200 万
个参数。
'''

'''
在编译和训练模型之前，一定要“冻结”卷积基。 冻结（ freeze）一个或多个层是指在训练
过程中保持其权重不变。如果不这么做，那么卷积基之前学到的表示将会在训练过程中被修改。
因为其上添加的 Dense 层是随机初始化的，所以非常大的权重更新将会在网络中传播，对之前
学到的表示造成很大破坏。
在 Keras 中，冻结网络的方法是将其 trainable 属性设为 False。
'''
print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))

# 利用冻结的卷积基端到端地训练模型
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import os

base_dir = r"E:\Code\DL_chap5\cats_vs_dogs\数据\cats_and_dogs_small"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

train_datagen = ImageDataGenerator(

    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
# 别忘了验证|测试数据不能增强
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc']
              )

history = model.fit_generator(
                    train_generator,
                    steps_per_epoch=100,
                    epochs=30,
                    validation_data=validation_generator,
                    validation_steps=50)

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
