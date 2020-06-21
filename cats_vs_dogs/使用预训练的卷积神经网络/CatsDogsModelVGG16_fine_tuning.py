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
from keras_preprocessing.image import ImageDataGenerator

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

conv_base.summary()
'''
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 150, 150, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________

Process finished with exit code 0

'''

# 我们将微调最后三个卷积层

'''
为什么不微调更多层？为什么不微调整个卷积基？你当然可以这么做，但需要考虑以下几点。
 卷积基中更靠底部的层编码的是更加通用的可复用特征，而更靠顶部的层编码的是更专
业化的特征。微调这些更专业化的特征更加有用，因为它们需要在你的新问题上改变用
途。微调更靠底部的层，得到的回报会更少。
 训练的参数越多，过拟合的风险越大。卷积基有 1500 万个参数，所以在你的小型数据
集上训练这么多参数是有风险的。
因此，在这种情况下，一个好策略是仅微调卷积基最后的两三层。我们从上一个例子结束
的地方开始，继续实现此方法。
'''
# 在卷积基上添加一个密集连接分类器
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# 微调模型

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

# 数据部分
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

# 展示结果
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
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
# val_loss: 0.1679 - val_acc: 0.9410
