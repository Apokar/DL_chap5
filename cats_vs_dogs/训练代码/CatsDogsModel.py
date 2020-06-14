# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 22:49
# @Author  : Apokar
# @Email   : Apokar@163.com
# @File    : CatsDogsModel.py
# @Comment : 构建网络

from keras import models
from keras import layers
def create_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64,(3,3),activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128,(3,3),activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128,(3,3),activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(512,activation='relu'))

    model.add(layers.Dense(1,activation='sigmoid'))


    from keras import optimizers

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss = 'binary_crossentropy',
                  metrics= ['acc'])


    return model

if __name__ == '__main__':
    model = create_model()
    model.summary()