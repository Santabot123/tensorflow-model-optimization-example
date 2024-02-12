import glob
import numpy as np
import os
import tempfile
import random
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import cv2
import tf_keras as keras

from tf_keras.utils import Sequence
from tf_keras import Sequential
from tf_keras.layers import Conv2D, Dense, MaxPooling2D, Input,BatchNormalization,Dropout,Flatten,ReLU

test_train_split=0.9


def save_tflite(model,name):
    with open(f'./models/tflite_{name}_model.tflite', 'wb') as f:
      f.write(model)

    model_size = os.path.getsize(f'./models/tflite_{name}_model.tflite')
    print(f"Tflite {name} tflite model size :", model_size , "bytes")



NORMAL_IMGS=[]
PNEUMONIA_IMGS=[]

NORMAL_IMGS+=glob.glob('/home/sasha/anaconda_projects/Pneumonia Detection/archive/chest_xray/test/NORMAL/*')
NORMAL_IMGS+=glob.glob('/home/sasha/anaconda_projects/Pneumonia Detection/archive/chest_xray/train/NORMAL/*')
NORMAL_IMGS+=glob.glob('/home/sasha/anaconda_projects/Pneumonia Detection/archive/chest_xray/val/NORMAL/*')

PNEUMONIA_IMGS+=glob.glob('/home/sasha/anaconda_projects/Pneumonia Detection/archive/chest_xray/test/PNEUMONIA/*')
PNEUMONIA_IMGS+=glob.glob('/home/sasha/anaconda_projects/Pneumonia Detection/archive/chest_xray/train/PNEUMONIA/*')
PNEUMONIA_IMGS+=glob.glob('/home/sasha/anaconda_projects/Pneumonia Detection/archive/chest_xray/val/PNEUMONIA/*')

# print(len(NORMAL_IMGS))
# print(len(PNEUMONIA_IMGS))



NORMAL_LABELS=np.zeros(len(NORMAL_IMGS))
PNEUMONIA_LABELS=np.ones(len(PNEUMONIA_IMGS))

ALL_IMGS=NORMAL_IMGS+PNEUMONIA_IMGS
ALL_LABLES=np.append(NORMAL_LABELS,PNEUMONIA_LABELS)

temp = list(zip(ALL_IMGS, ALL_LABLES))
random.shuffle(temp)
ALL_IMGS, ALL_LABLES= zip(*temp)

ALL_IMGS, ALL_LABLES= list(ALL_IMGS), list(ALL_LABLES)



x_names_train=ALL_IMGS[:int(np.floor(test_train_split*len(ALL_IMGS)))]
x_names_test=ALL_IMGS[int(np.floor(test_train_split*len(ALL_IMGS))):]

y_train=ALL_LABLES[:int(np.floor(test_train_split*len(ALL_LABLES)))]
y_test=ALL_LABLES[int(np.floor(test_train_split*len(ALL_LABLES))):]

x_names_train=np.array(x_names_train)
x_names_test=np.array(x_names_test)

y_train=np.array(y_train)
y_test=np.array(y_test)


class DataGenerator_with_load(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgs = []
        labels = self.y[inds]

        for x in self.x[inds]:
            img = cv2.imread(x)
            img = cv2.resize(img, [144, 144])
            img = img / 255
            imgs.append(img)

        imgs = np.array(imgs)

        batch_x = imgs
        batch_y = labels

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


train_gen = DataGenerator_with_load(x_names_train, y_train, 8)
test_gen = DataGenerator_with_load(x_names_test, y_test, 8)

def create_model():
    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', input_shape=(144, 144, 3)))
    model.add(ReLU())

    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, kernel_size=3, strides=1, padding='same'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=3, strides=1, padding='same'))
    model.add(ReLU())

    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    epochs = 2

    model.fit(train_gen,
              epochs=epochs,
              validation_data=test_gen,
              verbose=1)

    return model

model= create_model()

