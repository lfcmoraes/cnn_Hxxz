from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers.core import Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.metrics import categorical_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import ModelCheckpoint
from sklearn import metrics

import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
import cv2

plt.switch_backend('agg')

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'mean_squared_error' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'val_mean_squared_error' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training Mean Square Error (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation Mean Square Error (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Mean Square Error')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.savefig('mse.png')


def chart_regression(pred,y,sort=True):
    t = pd.DataFrame({'pred' : pred, 'y' : y.flatten()})
    if sort:
        t.sort_values(by=['y'],inplace=True)
    a = plt.plot(t['y'].tolist(),label='expected')
    b = plt.plot(t['pred'].tolist(),label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.title('RMSE ='+ str(np.round(100000*score)/100000))
    plt.savefig('predct.png')

dataset = pd.read_csv('data_set.csv')


#writing the columns of the csv in an list
X_1 = dataset.iloc[:,1].values
Y_1 = dataset.iloc[:,3].values

W = np.random.permutation(np.c_[X_1.reshape(len(X_1), -1), Y_1.reshape(len(Y_1), -1)])

X = W[:, :X_1.size//len(X_1)].reshape(X_1.shape)
Y = W[:, X_1.size//len(X_1):].reshape(Y_1.shape)

x_train = np.array([X[i] for i in range((len(X)//20),len(X))])
y_train = np.array([Y[i] for i in range((len(Y)//20),len(Y))])

x_test = np.array([X[i] for i in range((len(X)//20))])
y_test = np.array([Y[i] for i in range((len(Y)//20))])


df1 = pd.DataFrame(x_test)
df2 = pd.DataFrame(y_test)

df1.to_csv("x_test.csv")
df2.to_csv("y_test.csv")

train_data = []

load_train_data = []
for file in x_train:
        path = os.path.abspath('matrix/'+ file)
        H = np.load(path)
        load_train_data.append(H)
load_train_data = np.array(load_train_data)
train_data = load_train_data[:,:,:,np.newaxis]
train_data -= np.mean(train_data)
train_data /= np.std(train_data)

train_label = []
for file in y_train:
    train_label.append(file)
train_label = np.array(train_label)


test_data = []

load_test_data = []
for file in x_test:
        path = os.path.abspath('matrix/'+ file)
        H = np.load(path)
        load_test_data.append(H)
load_test_data = np.array(load_test_data)
test_data = load_test_data[:,:,:,np.newaxis]
test_data -= np.mean(test_data)
test_data /= np.std(test_data)

test_label = []
for file in y_test:
    test_label.append(file)
test_label = np.array(test_label)

model = Sequential()
model.add(Conv2D(64,(3,3), activation='relu', input_shape=(256,256,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512,(3,3),activation='relu'))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512,(3,3),activation='relu'))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='linear'))


model.compile(Adam(lr=.0001),loss='mean_squared_error', metrics=['mse'])


history = model.fit(train_data, train_label, epochs=5, batch_size=128, validation_split=0.2, verbose=1)

model.save('my_model.h5')
plot_history(history)

pred = model.predict(test_data, batch_size=128)
score = np.sqrt(metrics.mean_squared_error(pred,test_label))
print("Score (RMSE): {}".format(score))

chart_regression(pred.flatten(),test_label)





