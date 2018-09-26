from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers.core import Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.metrics import categorical_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *

import keras
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import shutil
import os
import cv2

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

train_path = 'matrix/train/'
val_path = 'matrix/val'
test_path = 'matrix/test/'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(256,256), classes=['l_n8','l_n6','l_n4','l_n2','l_0','l_2','l_4','l_6','l_8'],batch_size=128)
val_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(256,256), classes=['l_n8','l_n6','l_n4','l_n2','l_0','l_2','l_4','l_6','l_8'],batch_size=128)
test_baches = ImageDataGenerator().flow_from_directory(val_path, target_size=(256,256), classes=['l_n8','l_n6','l_n4','l_n2','l_0','l_2','l_4','l_6','l_8'],batch_size=128)


vgg16_model = keras.applications.vgg16.VGG16(weights=None,classes=9,input_shape=(256,256,3))

vgg16_model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

vgg16_model.fit_generator(train_batches, steps_per_epoch=60, validation_data= val_batches, validation_steps =15 , epochs=10, verbose=1)

model.save('Hxxz.h5')
