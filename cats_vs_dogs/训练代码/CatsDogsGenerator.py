# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 23:01
# @Author  : Apokar
# @Email   : Apokar@163.com
# @File    : CatsDogsGenerator.py
# @Comment : 数据预处理


from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from cats_vs_dogs.训练代码.CatsDogsModel import model


def data_generator():
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    # 所有图像乘以1/255缩放

    train_dir = r'E:\Code\DL_chap5\cats_vs_dogs\数据\cats_and_dogs_small\train'
    validation_dir = r'E:\Code\DL_chap5\cats_vs_dogs\数据\cats_and_dogs_small\validation'

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150), # 图像大小调整为150*150
        batch_size=20,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='binary'
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data = validation_generator,
        validation_steps=50
    )
    print("Saving model to disk \n")
    cats_dogs_model = "E:\Code\DL_chap5\cats_vs_dogs\模型文件\cats_dogs_model.h5"
    model.save(cats_dogs_model)

    import matplotlib.pyplot as plt

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)

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