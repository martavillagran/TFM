try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd
import random
from sklearn.model_selection import ShuffleSplit
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
# import tensorflow_probability as tfp
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.utils import to_categorical
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils_adam import FluidNetwork
from utils_adam import GA

def load_data_mnist():
  "Loads data and each time the function is called a new partition of xtrain is used"
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  # Shuffle data
  # rs = ShuffleSplit(n_splits=5, test_size=.1)
  # x_train, y_train  = rs.get_n_splits(X) pensar 
  # Normalize and transform to categorical
  # x_train = np.reshape(x_train, (x_train.shape[0], 784))/255.
  # x_test = np.reshape(x_test, (x_test.shape[0], 784))/255.
  # y_train = tf.keras.utils.to_categorical(y_train)
  # y_test = tf.keras.utils.to_categorical(y_test)
  # y_train = to_categorical(y_train)
  # y_test = to_categorical(y_test)
  X = np.vstack((x_train, x_test))
  y = np.hstack((y_train, y_test))
  x_train, x_test, y_train, y_test = train_test_split(X[1:2000], y[1:2000], test_size=0.5, stratify=y[1:2000])
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)
  x_test = np.reshape(x_test, (x_test.shape[0], 784))/255.
  x_train = x_train/255.
  return (x_train, y_train), (x_test, y_test)

def data_augmentation(x_train):
  # Data augmentation
  datagen = ImageDataGenerator(
      featurewise_center=False,
      featurewise_std_normalization=False,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      validation_split=0.2)
  # compute quantities required for featurewise normalization
  # (std, mean, and principal components if ZCA whitening is applied)
  datagen.fit(x_train)
  return datagen

(x_train, y_train), (x_test, y_test) = load_data_mnist()

x_train_aug = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], x_train.shape[2],1))

datagen = data_augmentation(x_train_aug)


# Topological parameters
window_size = 40
max_layers = 2+2 # con 5 capas al principio no funciona
num_max_units = 128
# input_dim = window_size
input_dim = 28*28
# output_dim = 2
output_dim = 10

layers = np.zeros(max_layers, dtype='uint32')
for i in range(max_layers): 
  if i == 0:
    layers[i] = input_dim
  elif i == max_layers-1:
    layers[i] = output_dim
  else:
    layers[i] = num_max_units

net = FluidNetwork(layers)

# Training parameters
batch_size = 128
epochs = 3000 # necesitaria algunas más
steps_per_epoch = int(x_train.shape[0]/batch_size)
lr = 1e-2
trigger = 0.01
print('Steps per epoch', steps_per_epoch)

# num_max_units = 128
# num_min_units = 10
# num_max_layers = 5
# num_min_layers = 2
# input_dim = net.num_features
# output_dim = net.num_classes
# n_stall_generations = 10
# n_iters = 20
# sample_update = 0.1
# # Invoke GA class 
# net_algo = net.copy_actual_instance()
# algo = GA(net_algo, sample_update, num_max_units, num_min_units, num_max_layers, num_min_layers, input_dim, output_dim, datagen)
# # Run GA optimization
# algo.run_algo(n_stall_generations, n_iters)



history = net.train(x_train, y_train, x_test, y_test, epochs, datagen,
    batch_size, lr, trigger)

# # Save model
# import os
# dir = os.path.dirname(os.path.abspath('adam.py'))

# f = open('C:\\Users\\marta\\OneDrive - Universidad Pontificia Comillas\\MASTER IA', 'wb')
# pickle.dump(net, f)
# f.close()

# plt.figure()
# plt.plot(history['val_acc'])

# plt.title('Accuracy para el MNIST con una red evolutiva')

# plt.figure()
# plt.plot(history['train_loss'])
# plt.plot(history['val_loss'])
# plt.legend(['Train loss', 'Val loss'])

# plt.title('Loss para el MNIST con una red evolutiva')

# # Check convergencia
# plt.figure()
# plt.plot(history['train_loss'][2500:3000])
# plt.plot(history['val_loss'][2500:3000])
# plt.legend(['Train loss', 'Val loss'])

# plt.title('Comprobación convergencia del loss para el MNIST con una red evolutiva')

# # Analisis del trigger
# aux = np.diff(np.log(np.array(history['val_loss'])))
# plt.figure()
# plt.plot(aux[5:])
# plt.title('Diferencia del loss en cada epoch')

# # Zoom a la imagen anterior
# plt.figure()
# plt.plot(aux[600:700])
# # plt.legend(['Diferencia del loss', 'Val loss'])
# plt.title('Diferencia del loss en cada epoch')

# plt.figure()
# plt.plot(history['val_loss'][600:700])
# plt.title('Diferencia del loss en cada epoch')
