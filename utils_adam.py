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
import copy 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
  x_train, x_test, y_train, y_test = train_test_split(X[1:2000], y[1:2000], test_size=0.3, stratify=y[1:2000])
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)
  x_test = np.reshape(x_test, (x_test.shape[0], 784))/255.
  x_train = x_train/255.
  return (x_train, y_train), (x_test, y_test)
class FluidNetwork:
#-------------------------------------------------------------------------------    
  def __init__(self, layers):
      self.layers = layers
      self.L = len(layers) # input, hidden & output layer
      self.num_features = layers[0]
      self.num_classes = layers[-1]
      self.adam = 1
      
      self.W = {}
      self.b = {}
      
      self.prev_W = {}
      self.prev_b = {}

      self.dW = {}
      self.db = {}
      # Adam optimizer
      self.m_dW = {}
      self.v_dW = {}
      self.m_db = {}
      self.v_db = {}
      self.t = 0
      self.prev_m_dW = {}
      self.prev_v_dW = {}
      self.prev_m_db = {}
      self.prev_v_db = {}

      self.setup()
      self.new_topology = []
#-------------------------------------------------------------------------------
  def setup(self): 
      for i in range(1, self.L):
        self.W[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) # to enable future modifications
        self.b[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
        self.W[i].assign(tf.Variable(tf.random.normal(shape=(self.layers[i],self.layers[i-1]))))
        self.b[i].assign(tf.Variable(tf.random.normal(shape=(self.layers[i],1))))
        if self.adam == 1:
          self.m_dW[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) # to enable future modifications
          self.m_db[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
          self.m_dW[i].assign(tf.Variable(tf.zeros(shape=(self.layers[i],self.layers[i-1]))))
          self.m_db[i].assign(tf.Variable(tf.zeros(shape=(self.layers[i],1))))
          self.v_dW[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) # to enable future modifications
          self.v_db[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
          self.v_dW[i].assign(tf.Variable(tf.zeros(shape=(self.layers[i],self.layers[i-1]))))
          self.v_db[i].assign(tf.Variable(tf.zeros(shape=(self.layers[i],1))))
#-------------------------------------------------------------------------------
  def forward_pass(self, X):
      try:
        A = tf.convert_to_tensor(X, dtype=tf.float32)
        for i in range(1, self.L):
            Z = tf.matmul(A,tf.transpose(self.W[i])) + tf.transpose(self.b[i])
            if i != self.L-1:
                A = tf.nn.relu(Z)
            else:
                A = Z
      except:
        print('Next gen')
        A = []
      return A
#-------------------------------------------------------------------------------
  def compute_loss(self, A, Y):
    if len(A) == 0 :
      loss = np.inf
    else:
      loss = tf.nn.softmax_cross_entropy_with_logits(Y,A)
      # loss = tf.nn.BinaryCrossentropy(Y,A)
      loss = tf.reduce_mean(loss)
    return loss
#-------------------------------------------------------------------------------   
  def update_params(self, lr, adam): # adam = 1 to perform adam optimizer
    if adam == 1:
          # Adams parameters
          beta1 = 0.9
          beta2 = 0.999
          epsilon = 1e-8
    for i in range(1,self.L):
      if adam == 0:
        self.W[i].assign_sub(lr * self.dW[i])
        self.b[i].assign_sub(lr * self.db[i])
      else:
        # Momentum beta1
        # W
        self.m_dW[i].assign(beta1*self.m_dW[i] + (1-beta1)*self.dW[i])
        # b
        self.m_db[i].assign(beta1*self.m_db[i] + (1-beta1)*self.db[i])
        # RMS beta2
        # W
        self.v_dW[i].assign(beta2*self.v_dW[i] + (1-beta2)*tf.math.square(self.dW[i]))
        # b
        self.v_db[i].assign(beta2*self.v_db[i] + (1-beta2)*tf.math.square(self.db[i]))
        # Bias correction
        m_dW_corr = self.m_dW[i]/(1-beta1**self.t)
        m_db_corr = self.m_db[i]/(1-beta1**self.t)
        v_dW_corr = self.v_dW[i]/(1-beta2**self.t)
        v_db_corr = self.v_db[i]/(1-beta2**self.t)
        # Update weights and biases
        self.W[i].assign_sub(lr*tf.math.divide_no_nan(m_dW_corr, tf.math.sqrt(v_dW_corr)+epsilon))
        self.b[i].assign_sub(lr*tf.math.divide_no_nan(m_db_corr, tf.math.sqrt(v_db_corr)+epsilon))
#-------------------------------------------------------------------------------          
  def predict(self, X):
      
      A = self.forward_pass(X)
      return tf.argmax(tf.nn.softmax(A), axis=1)
      # return tf.argmax(tf.nn.sigmoid(A), axis=1)
#-------------------------------------------------------------------------------  
  def info(self):
      num_params = 0
      for i in range(1, self.L):
          num_params += self.W[i].numpy().shape[0] * self.W[i].numpy().shape[1]
          num_params += self.b[i].numpy().shape[0]
      print('Input Features:', self.num_features)
      print('Number of Classes:', self.num_classes)
      print('Hidden Layers:')
      print('--------------')
      for i in range(1, self.L-1):
          print('Layer {}, Units {}'.format(i, self.layers[i]))
      print('--------------')
      print('Number of parameters:', num_params)
#-------------------------------------------------------------------------------
  def train_on_batch(self, X, Y, lr):
        
      X = tf.convert_to_tensor(X, dtype=tf.float32)
      Y = tf.convert_to_tensor(Y, dtype=tf.float32)
        
      with tf.GradientTape(persistent=True) as tape:
          A = self.forward_pass(X)
          loss = self.compute_loss(A, Y)
      if len(A) >0:
        for i in range(1, self.L):
            self.dW[i] = tape.gradient(loss, self.W[i])
            self.db[i] = tape.gradient(loss, self.b[i])
        del tape
        self.update_params(lr, self.adam)
        flag = 0
        loss = loss.numpy()
      else:
        flag = -1
      return loss, flag # flag -1 stop that NN
#-------------------------------------------------------------------------------    
  def update_keys(self):
    ini_list = {}
    for i in range(1, self.L): ini_list[i] = i
    self.b = dict(zip(ini_list, list(self.b.values())))    
    self.W = dict(zip(ini_list, list(self.W.values())))  
    if self.adam == 1:
        self.m_db = dict(zip(ini_list, list(self.m_db.values())))    
        self.m_dW = dict(zip(ini_list, list(self.m_dW.values())))  
        self.v_db = dict(zip(ini_list, list(self.v_db.values())))    
        self.v_dW = dict(zip(ini_list, list(self.v_dW.values())))  
#-------------------------------------------------------------------------------  
  def sample_distribution(self,param, layer, shape):
    'Param: W (1) /b (0), layer: number of weight, size: tensor shape'
    # Find best distribution to fit in dataset
    if param == 1:
      dataset = pd.DataFrame(self.prev_W[layer].numpy())
    else:
      dataset = pd.DataFrame(self.prev_b[layer].numpy())

    f = Fitter(dataset,distributions= get_common_distributions()) # only common distribution out of the 80 are tested
    f.fit()
    summary = pd.DataFrame(f.summary(method='bic')) 
    dist_name = summary['bic'].keys()[0] # best dist according to bic values
    # best_params = list(f.get_best(method='bic').values())[0] # best params 
    best_params = f.fitted_param[dist_name]
    # Invoke distribution object
    tfd = tfp.distributions
    # Switch-case sentence
    if dist_name == 'cauchy':
      return tf.cast(tf.Variable(tfd.Cauchy(loc=best_params[0], scale=best_params[1]).sample(shape)), tf.float32)
    elif dist_name == 'chi2':
      return tf.cast(tf.Variable(tfd.Chi2(best_params[0]).sample(shape)), tf.float32)
    elif dist_name == 'expon':
      return tf.cast(tf.Variable(tfd.Exponential(np.abs(best_params[0])).sample(shape)), tf.float32)
    elif dist_name == 'exponpow':
      return tf.cast(tf.Variable(tfd.ExponentiallyModifiedGaussian(loc=best_params[0],
                                              scale=best_params[1],
                                              rate=best_params[2]).sample(shape)), tf.float32)
    elif dist_name == 'gamma':
      return tf.cast(tf.Variable(tfd.Gamma(best_params[0], best_params[1], best_params[2]).sample(shape)), tf.float32)
    elif dist_name == 'lognorm':
      return tf.cast(tf.Variable(tfd.LogNormal(loc=best_params[0], scale=best_params[1]).sample(shape)), tf.float32)
    elif dist_name == 'norm':
      return tf.cast(tf.Variable(tfd.Normal(loc=best_params[0], scale=best_params[1]).sample(shape)), tf.float32)
    elif dist_name == 'powerlaw': # Its not defined
      return tf.cast(tf.Variable(tfd.Normal(loc=best_params[0], scale=best_params[1]).sample(shape)), tf.float32)
    elif dist_name == 'rayleigh':
      return tf.cast(tf.Variable(tfp.random.rayleigh(scale=best_params[1]).sample(shape)), tf.float32)
    else: # uniform
      return tf.cast(tf.Variable(tfd.Uniform(low=best_params[0], high=best_params[1]).sample(shape)), tf.float32)
#-------------------------------------------------------------------------------
  def activate_neurons(self, i, flag):
    "Activate one neuron means adding a extra cols for W[i] and extra rows for W[i-1]"
    if self.layers[i-1] < self.new_topology[i-1]: # activate more neurons
      neurons_to_act = self.new_topology[i-1] - self.layers[i-1]
      # First add cols to W[i]
      # W
      if flag == 1:
        try:
          shape = (self.layers[i], neurons_to_act)
          aux = self.sample_distribution(1, i, shape)
        except:
          print('Normal dist instead')
        finally:
          aux = tf.Variable(tf.random.normal(shape=(self.layers[i], neurons_to_act)))
          aux2 = tf.Variable(tf.zeros(shape=(self.layers[i], neurons_to_act)))
      else:
        aux = tf.Variable(tf.random.normal(shape=(self.layers[i], neurons_to_act)))
        aux2 = tf.Variable(tf.zeros(shape=(self.layers[i], neurons_to_act)))
      self.W[i].assign(tf.concat(axis=1, values=[self.W[i], aux]))
      if self.adam:
        self.m_dW[i].assign(tf.concat(axis=1, values=[self.m_dW[i], aux2]))
        self.v_dW[i].assign(tf.concat(axis=1, values=[self.v_dW[i], aux2]))
      # Then add rows to W[i-1]
      # W
      if flag == 1:
        try:
          shape = (neurons_to_act, self.new_topology[i-2])
          aux = self.sample_distribution(1, i, shape)
        except:
          print('Normal dist instead')
        finally:
          aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, self.new_topology[i-2])))
          aux2 = tf.Variable(tf.zeros(shape=(neurons_to_act, self.new_topology[i-2])))
      else:
        aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, self.new_topology[i-2])))
        aux2 = tf.Variable(tf.zeros(shape=(neurons_to_act, self.new_topology[i-2])))
      self.W[i-1].assign(tf.concat(axis=0, values=[self.W[i-1], aux]))
      if self.adam:
        self.m_dW[i-1].assign(tf.concat(axis=0, values=[self.m_dW[i-1], aux2]))
        self.v_dW[i-1].assign(tf.concat(axis=0, values=[self.v_dW[i-1], aux2]))
      # b
      if flag == 1:
        try:
          shape = (neurons_to_act, 1)
          aux = self.sample_distribution(0, i, shape)
        except:
          print('Normal dist instead')
        finally:
          aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, 1)))
          aux2 = tf.Variable(tf.zeros(shape=(neurons_to_act, 1)))
      else:
        aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, 1)))
        aux2 = tf.Variable(tf.zeros(shape=(neurons_to_act, 1)))
      self.b[i-1].assign(tf.concat(axis=0, values=[self.b[i-1], aux]))
      if self.adam:
        self.m_db[i-1].assign(tf.concat(axis=0, values=[self.m_db[i-1], aux2]))
        self.v_db[i-1].assign(tf.concat(axis=0, values=[self.v_db[i-1], aux2]))

    elif self.layers[i-1] > self.new_topology[i-1]: # deactivate some randomly neurons
      neurons_to_act = self.new_topology[i-1]
      random_index = np.random.choice(self.layers[i-1], neurons_to_act, replace=False)
      # First remove extra cols from the current layer
      # W
      aux = pd.DataFrame(self.W[i].numpy()).copy()
      aux = aux.reindex(columns=random_index).dropna(axis=1)
      self.W[i].assign(aux.to_numpy())
      if self.adam:
        aux2 = pd.DataFrame(self.m_dW[i].numpy()).copy()
        aux2 = aux2.reindex(columns=random_index).dropna(axis=1)
        aux3 = pd.DataFrame(self.v_dW[i].numpy()).copy()
        aux3 = aux3.reindex(columns=random_index).dropna(axis=1)   
        self.m_dW[i].assign(aux2.to_numpy())
        self.v_dW[i].assign(aux3.to_numpy())
      # Then remove extra rows from the previous layer
      # W
      aux = np.transpose(pd.DataFrame(self.W[i-1].numpy())).copy()
      aux = aux.reindex(columns=random_index).dropna(axis=1)
      self.W[i-1].assign(aux.T.to_numpy())
      if self.adam:
        aux2 = np.transpose(pd.DataFrame(self.m_dW[i-1].numpy())).copy()
        aux2 = aux2.reindex(columns=random_index).dropna(axis=1)
        aux3 = np.transpose(pd.DataFrame(self.v_dW[i-1].numpy())).copy()
        aux3 = aux3.reindex(columns=random_index).dropna(axis=1) 
        self.m_dW[i-1].assign(aux2.T.to_numpy())
        self.v_dW[i-1].assign(aux3.T.to_numpy())
      # b
      aux = np.transpose(pd.DataFrame(self.b[i-1].numpy())).copy()
      aux = aux.reindex(columns=random_index).dropna(axis=1)
      self.b[i-1].assign(aux.T.to_numpy())
      if self.adam:
        aux2 = np.transpose(pd.DataFrame(self.m_db[i-1].numpy())).copy()
        aux2 = aux2.reindex(columns=random_index).dropna(axis=1)
        aux3 = np.transpose(pd.DataFrame(self.v_db[i-1].numpy())).copy()
        aux3 = aux3.reindex(columns=random_index).dropna(axis=1)
        self.m_db[i-1].assign(aux2.T.to_numpy())
        self.v_db[i-1].assign(aux3.T.to_numpy())
#-------------------------------------------------------------------------------    
  def AG_update(self, flag): # weight padding # i is the index for weights not for the layers 
    # self.prev_W = self.W.copy()
    # self.prev_b = self.b.copy()
    # self.prev_m_dW = self.m_dW.copy()
    # self.prev_v_dW = self.v_dW.copy()
    # self.prev_m_db = self.m_db.copy()
    # self.prev_v_db = self.v_db.copy()
    
    self.prev_W = copy.deepcopy(self.W)
    self.prev_b = copy.deepcopy(self.b)
    self.prev_m_dW = copy.deepcopy(self.m_dW)
    self.prev_v_dW = copy.deepcopy(self.v_dW)
    self.prev_m_db = copy.deepcopy(self.m_db)
    self.prev_v_db = copy.deepcopy(self.v_db)
    new_L = len(self.new_topology)

    for i in range(2, new_L): # 2 as the input alwyas remain the same
      if self.L < new_L : # The new topology contains more layers
        if i <= self.L - 2 : # Only apply for hidden layers
          self.activate_neurons(i, flag)
        else: 
          if i == new_L - 1: # copy the output weights from the previous structure
            self.W[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) # to enable future modifications
            self.b[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
            if self.adam:
              self.m_dW[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
              self.v_dW[i] = tf.Variable(1.0, shape=tf.TensorShape(None))    
              self.m_db[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
              self.v_db[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
            if (self.layers[-2] >= self.new_topology[-2]):
              neurons_to_act = self.new_topology[i-1]
              random_index = np.random.choice(self.layers[self.L-2], neurons_to_act, replace=False)
              aux = pd.DataFrame(self.prev_W[self.L -1].numpy()).copy()
              aux = aux.reindex(columns=random_index).dropna(axis=1)
              self.W[i].assign(aux.to_numpy())
              if self.adam:
                aux2 = pd.DataFrame(self.prev_m_dW[self.L -1].numpy()).copy()
                aux2 = aux2.reindex(columns=random_index).dropna(axis=1)
                aux3 = pd.DataFrame(self.prev_v_dW[self.L -1].numpy()).copy()
                aux3 = aux3.reindex(columns=random_index).dropna(axis=1)
                self.m_dW[i].assign(aux2.to_numpy())
                self.v_dW[i].assign(aux3.to_numpy())
            else:
              neurons_to_act = self.new_topology[-2] - self.layers[-2]
              if flag == 1: # cambiarlo a 1
                shape = (self.new_topology[i], neurons_to_act)
                try:
                  aux = self.sample_distribution(1, self.L-1, shape)
                except:
                  print('Normal dist instead')
                  aux = tf.Variable(tf.random.normal(shape=(self.new_topology[i], neurons_to_act)))
                  aux2 = tf.Variable(tf.zeros(shape=(self.new_topology[i], neurons_to_act)))
               # finally:
                   # aux = tf.Variable(tf.random.normal(shape=(self.new_topology[i], neurons_to_act)))
              else:
                 aux = tf.Variable(tf.random.normal(shape=(self.new_topology[i], neurons_to_act)))
                 aux2 = tf.Variable(tf.zeros(shape=(self.new_topology[i], neurons_to_act)))
              self.W[i].assign(tf.concat(axis=1, values=[self.prev_W[self.L -1], aux]))
              if self.adam:
                self.m_dW[i].assign(tf.concat(axis=1, values=[self.prev_m_dW[self.L -1], aux2]))
                self.v_dW[i].assign(tf.concat(axis=1, values=[self.prev_v_dW[self.L -1], aux2]))
            self.b[i].assign(self.prev_b[self.L -1])
            if self.adam:
              self.m_db[i].assign(self.m_db[self.L -1])
              self.v_db[i].assign(self.v_db[self.L -1])

          else: # add more layers and reestructure weights for the previous layer
          # Reestructure weights for the last layer
            if i == self.L -1 :
              if self.layers[i-1] < self.new_topology[i-1]: # activate more neurons
                neurons_to_act = self.new_topology[i-1] - self.layers[i-1]
                # Then add rows to W[i-1]
                # W
                if flag == 1:
                  try:
                    shape = (neurons_to_act, self.new_topology[i-2])
                    aux = self.sample_distribution(1, i, shape)
                  except:
                    print('Normal dist instead')
                    aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, self.new_topology[i-2])))
                    aux2 = tf.Variable(tf.zeros(shape=(neurons_to_act, self.new_topology[i-2])))
                 # finally:
                    # aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, self.new_topology[i-2])))
                else:
                  aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, self.new_topology[i-2])))
                  aux2 = tf.Variable(tf.zeros(shape=(neurons_to_act, self.new_topology[i-2])))
                self.W[i-1].assign(tf.concat(axis=0, values=[self.W[i-1], aux]))
                if self.adam:
                  self.m_dW[i-1].assign(tf.concat(axis=0, values=[self.m_dW[i-1], aux2]))
                  self.v_dW[i-1].assign(tf.concat(axis=0, values=[self.v_dW[i-1], aux2]))
                # b
                if flag == 1:
                  try:
                    shape = (neurons_to_act, 1)
                    aux = tf.cast(self.sample_distribution(0, i, shape), tf.float32)
                  except:
                    print('Normal distribution')
                    aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, 1)))
                    aux2 = tf.Variable(tf.zeros(shape=(neurons_to_act, 1)))
                  # finally:
                   # aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, 1)))
                else:
                  aux2 = tf.Variable(tf.random.normal(shape=(neurons_to_act, 1)))
                  aux = tf.Variable(tf.zeros(shape=(neurons_to_act, 1)))
                self.b[i-1].assign(tf.concat(axis=0, values=[self.b[i-1], aux]))
                if self.adam:
                  self.m_db[i-1].assign(tf.concat(axis=0, values=[self.m_db[i-1], aux2]))
                  self.v_db[i-1].assign(tf.concat(axis=0, values=[self.v_db[i-1], aux2]))
              elif self.layers[i-1] > self.new_topology[i-1]: # deactivate some randomly neurons
                neurons_to_act = self.new_topology[i-1]
                random_index = np.random.choice(self.layers[i-1], neurons_to_act, replace=False)
                # Then remove extra rows from the previous layer
                # W
                aux = np.transpose(pd.DataFrame(self.W[i-1].numpy())).copy()
                aux = aux.reindex(columns=random_index).dropna(axis=1)
                self.W[i-1].assign(aux.T.to_numpy())
                if self.adam:
                  aux2 = np.transpose(pd.DataFrame(self.m_dW[i-1].numpy())).copy()
                  aux2 = aux2.reindex(columns=random_index).dropna(axis=1)
                  aux3 = np.transpose(pd.DataFrame(self.v_dW[i-1].numpy())).copy()
                  aux3 = aux3.reindex(columns=random_index).dropna(axis=1)
                  self.m_dW[i-1].assign(aux2.T.to_numpy())
                  self.v_dW[i-1].assign(aux3.T.to_numpy())
                # b
                aux = np.transpose(pd.DataFrame(self.b[i-1].numpy())).copy()
                aux = aux.reindex(columns=random_index).dropna(axis=1)
                self.b[i-1].assign(aux.T.to_numpy())
                if self.adam:
                  aux2 = np.transpose(pd.DataFrame(self.m_db[i-1].numpy())).copy()
                  aux2 = aux2.reindex(columns=random_index).dropna(axis=1)
                  aux3 = np.transpose(pd.DataFrame(self.v_db[i-1].numpy())).copy()
                  aux3 = aux.reindex(columns=random_index).dropna(axis=1)
                  self.m_db[i-1].assign(aux2.T.to_numpy())
                  self.v_db[i-1].assign(aux3.T.to_numpy())
          if new_L != i -1:            
              # Add more layers
              self.W[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) # to enable future modifications
              self.b[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
              if self.adam:
                self.m_dW[i] = tf.Variable(1.0, shape=tf.TensorShape(None))
                self.v_dW[i] = tf.Variable(1.0, shape=tf.TensorShape(None))
                self.m_db[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
                self.v_db[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
              if flag == 1: # cambiarlo a 1
                try:
                  self.W[i].assign(self.sample_distribution(1, self.L-1, shape=(self.new_topology[i],self.new_topology[i-1])))
                  self.b[i].assign(tf.cast(self.sample_distribution(0, self.L-1, shape=(self.new_topology[i],1)), tf.float32))
                except:
                  print('Normal dist instead')
                  self.W[i].assign(tf.Variable(tf.random.normal(shape=(self.new_topology[i],self.new_topology[i-1]))))
                  self.b[i].assign(tf.Variable(tf.zeros(shape=(self.new_topology[i],1)))) 
                  if self.adam:
                    self.m_dW[i].assign(tf.Variable(tf.zeros(shape=(self.new_topology[i],self.new_topology[i-1]))))
                    self.v_dW[i].assign(tf.Variable(tf.zeros(shape=(self.new_topology[i],self.new_topology[i-1]))))
                    self.m_db[i].assign(tf.Variable(tf.zeros(shape=(self.new_topology[i],1)))) 
                    self.v_db[i].assign(tf.Variable(tf.zeros(shape=(self.new_topology[i],1)))) 
                # finally:
                 # self.W[i].assign(tf.Variable(tf.random.normal(shape=(self.new_topology[i],self.new_topology[i-1]))))
                 # self.b[i].assign(tf.Variable(tf.random.normal(shape=(self.new_topology[i],1)))) 
              else:
                self.W[i].assign(tf.Variable(tf.random.normal(shape=(self.new_topology[i],self.new_topology[i-1]))))
                self.b[i].assign(tf.Variable(tf.random.normal(shape=(self.new_topology[i],1)))) 
                if self.adam:
                  self.m_dW[i].assign(tf.Variable(tf.zeros(shape=(self.new_topology[i],self.new_topology[i-1]))))
                  self.v_dW[i].assign(tf.Variable(tf.zeros(shape=(self.new_topology[i],self.new_topology[i-1]))))
                  self.m_db[i].assign(tf.Variable(tf.zeros(shape=(self.new_topology[i],1)))) 
                  self.v_db[i].assign(tf.Variable(tf.zeros(shape=(self.new_topology[i],1)))) 

      elif self.L > new_L: # The new topology contains less layers
        if i == new_L - 1 :
        # Firstly, add/remove weights from the previous layer
          if self.layers[i-1] < self.new_topology[i-1]: # activate more neurons
            neurons_to_act = self.new_topology[i-1] - self.layers[i-1]
            # Then add rows to W[i-1]
            # W
            if flag == 1:
              try:
                shape = (neurons_to_act, self.new_topology[i-2])
                aux = self.sample_distribution(1, i, shape)
              except:
                print('Normal dist instead')
                aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, self.new_topology[i-2])))
                aux2 = tf.Variable(tf.zeros(shape=(neurons_to_act, self.new_topology[i-2])))
              # finally:
               # aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, self.new_topology[i-2])))
            else:
              aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, self.new_topology[i-2])))
              aux2 = tf.Variable(tf.zeros(shape=(neurons_to_act, self.new_topology[i-2])))
            self.W[i-1].assign(tf.concat(axis=0, values=[self.W[i-1], aux]))
            if self.adam:
              self.m_dW[i-1].assign(tf.concat(axis=0, values=[self.m_dW[i-1], aux2]))
              self.v_dW[i-1].assign(tf.concat(axis=0, values=[self.v_dW[i-1], aux2]))
            # b
            if flag == 1:
              try:
                shape = (neurons_to_act, 1)
                aux = tf.cast(self.sample_distribution(0, i, shape), tf.float32)
              except:
                print('Normal dist instead')
                aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, 1)))
                aux2 = tf.Variable(tf.random.normal(shape=(neurons_to_act, 1)))
              # finally:
               # aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, 1)))
            else:
              aux = tf.Variable(tf.random.normal(shape=(neurons_to_act, 1)))
              aux2 = tf.Variable(tf.zeros(shape=(neurons_to_act, 1)))
            self.b[i-1].assign(tf.concat(axis=0, values=[self.b[i-1], aux]))
            if self.adam:
              self.m_db[i-1].assign(tf.concat(axis=0, values=[self.m_db[i-1], aux2]))
              self.v_db[i-1].assign(tf.concat(axis=0, values=[self.v_db[i-1], aux2]))

          elif self.layers[i-1] > self.new_topology[i-1]: # deactivate some randomly neurons
            neurons_to_act = self.new_topology[i-1]
            random_index = np.random.choice(self.layers[i-1], neurons_to_act, replace=False)
            # Then remove extra rows from the previous layer
            # W
            aux = np.transpose(pd.DataFrame(self.W[i-1].numpy())).copy()
            aux = aux.reindex(columns=random_index).dropna(axis=1)
            self.W[i-1].assign(aux.T.to_numpy())
            if self.adam:
              aux2 = np.transpose(pd.DataFrame(self.m_dW[i-1].numpy())).copy()
              aux2 = aux2.reindex(columns=random_index).dropna(axis=1)
              self.m_dW[i-1].assign(aux2.T.to_numpy())
              aux3 = np.transpose(pd.DataFrame(self.v_dW[i-1].numpy())).copy()
              aux3 = aux3.reindex(columns=random_index).dropna(axis=1)
              self.v_dW[i-1].assign(aux3.T.to_numpy())
            # b
            aux = np.transpose(pd.DataFrame(self.b[i-1].numpy())).copy()
            aux = aux.reindex(columns=random_index).dropna(axis=1)
            self.b[i-1].assign(aux.T.to_numpy())
            if self.adam:
              aux2 = np.transpose(pd.DataFrame(self.m_db[i-1].numpy())).copy()
              aux2 = aux2.reindex(columns=random_index).dropna(axis=1)
              self.m_db[i-1].assign(aux2.T.to_numpy())
              aux3 = np.transpose(pd.DataFrame(self.v_db[i-1].numpy())).copy()
              aux3 = aux3.reindex(columns=random_index).dropna(axis=1)
              self.v_db[i-1].assign(aux3.T.to_numpy())

          # Secondly, remove extra layers and then assign the ouput weights to the output layer
          index_aux = np.arange(start=1, stop=new_L-1, step=1)
          # index_aux[-1] = self.L - 1
          mask = np.isin(list(self.W.keys()), index_aux) == False
          keys_to_remove = np.array(list(self.W.keys()))[mask]
          # W
          d = self.W
          l = keys_to_remove
          list(map(d.__delitem__, filter(d.__contains__,l)))
          # b
          d = self.b
          l = keys_to_remove
          list(map(d.__delitem__, filter(d.__contains__,l)))
          if self.adam:
            # Adam momentum
            # mdW
            d = self.m_dW
            l = keys_to_remove
            list(map(d.__delitem__, filter(d.__contains__,l)))
            # mdb
            d = self.m_db
            l = keys_to_remove
            list(map(d.__delitem__, filter(d.__contains__,l)))
            # vdW
            d = self.v_dW
            l = keys_to_remove
            list(map(d.__delitem__, filter(d.__contains__,l)))
            # vdb
            d = self.v_db
            l = keys_to_remove
            list(map(d.__delitem__, filter(d.__contains__,l)))
          # Assign output weights to the ouput layer
          self.W[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) # to enable future modifications
          self.b[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
          if self.adam:
            self.m_dW[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
            self.v_dW[i] = tf.Variable(1.0, shape=tf.TensorShape(None))    
            self.m_db[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
            self.v_db[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
          if (self.layers[-2] >= self.new_topology[-2]):
            neurons_to_act = self.new_topology[i-1]
            random_index = np.random.choice(self.layers[self.L-2], neurons_to_act, replace=False)
            aux = pd.DataFrame(self.prev_W[self.L -1].numpy()).copy()
            aux = aux.reindex(columns=random_index).dropna(axis=1)
            self.W[i].assign(aux.to_numpy())
            if self.adam:
              aux2 = pd.DataFrame(self.prev_m_dW[self.L -1].numpy()).copy()
              aux2 = aux2.reindex(columns=random_index).dropna(axis=1)
              self.m_dW[i].assign(aux2.to_numpy())
              aux3 = pd.DataFrame(self.prev_v_dW[self.L -1].numpy()).copy()
              aux3 = aux3.reindex(columns=random_index).dropna(axis=1)
              self.v_dW[i].assign(aux3.to_numpy())
          else:
            neurons_to_act = self.new_topology[-2] - self.layers[-2]
            if flag == 1:
              try:
                shape = (self.new_topology[i], neurons_to_act)
                aux = self.sample_distribution(1, i, shape)
              except:
                print('Normal dist instead')
                aux = tf.Variable(tf.random.normal(shape=(self.new_topology[i], neurons_to_act)))
                aux2 = tf.Variable(tf.zeros(shape=(self.new_topology[i], neurons_to_act)))
              # finally:
               # aux = tf.Variable(tf.random.normal(shape=(self.new_topology[i], neurons_to_act)))
            else:
              aux = tf.Variable(tf.random.normal(shape=(self.new_topology[i], neurons_to_act)))
              aux2 = tf.Variable(tf.zeros(shape=(self.new_topology[i], neurons_to_act)))
            self.W[i].assign(tf.concat(axis=1, values=[self.prev_W[self.L -1], aux]))
            if self.adam:
              self.m_dW[i].assign(tf.concat(axis=1, values=[self.prev_m_dW[self.L -1], aux2]))
              self.v_dW[i].assign(tf.concat(axis=1, values=[self.prev_v_dW[self.L -1], aux2]))

          self.b[i].assign(self.prev_b[self.L -1])    
          if self.adam:    
            self.v_db[i].assign(self.prev_v_db[self.L -1])  
            self.m_db[i].assign(self.prev_m_db[self.L -1])     
          break
        else:
        # Only apply for hidden layers
          self.activate_neurons(i, flag)
      else: # The new topology contains the same layers but might differ in the number of units per layer
        self.activate_neurons(i, flag)

    # Update layers attributes and keys
    self.L = len(self.new_topology)
    self.layers = self.new_topology
#-------------------------------------------------------------------------------
  def train(self, x_train, y_train, x_test, y_test, epochs, datagen, batch_size, lr, trigger):
      trigger_ag = np.array([100, 500, 1000, 2000, 3000])
      history = {
          'val_loss':[],
          'train_loss':[],
          'val_acc':[],
          'val_f1':[],
      }
      flag2 = 1
      for e in range(0, epochs):
          if np.sum(e==trigger_ag)>0:
              flag2 = 0
          self.t = e +1 
          epoch_train_loss = 0.
          print('Epoch{}'.format(e), end='.')
          batches = 0
          steps_per_epoch = int(x_train.shape[0]/batch_size)
          x_train_aug = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], x_train.shape[2],1))
          for x_batch_aux, y_batch in datagen.flow(x_train_aug, y_train, batch_size=batch_size):
              x_batch = np.reshape(x_batch_aux, (x_batch_aux.shape[0], x_batch_aux.shape[1]**2))
              # x_batch = x_train[i*batch_size:(i+1)*batch_size]
              # y_batch = y_train[i*batch_size:(i+1)*batch_size]
              batches +=1
              batch_loss, flag = self.train_on_batch(x_batch, y_batch,lr)
              if flag == -1:
                # print('Next epoch')
                break
              epoch_train_loss += batch_loss
              if batches%int(steps_per_epoch/1) == 0:
                print(end='.')
              if batches >=len(x_train)/batch_size: # we need to break the loop by hand because the generator loops indefinitely
                # print('Next epoch, flag2',flag2)
                break
          history['train_loss'].append(epoch_train_loss/steps_per_epoch)
          val_A = self.forward_pass(x_test)  # Como los datos van a estar desbalanceados utilizar mejor otra métrica SMOTE
          val_loss = self.compute_loss(val_A, y_test)
          if np.isfinite(val_loss):
              val_loss = val_loss.numpy()
          history['val_loss'].append(val_loss)
          val_preds = self.predict(x_test)
          y_test_aux = pd.DataFrame(y_test).idxmax(axis=1).values
          # f1_score_val = f1_score(val_preds, y_test_aux)
          # val_acc = accuracy_score(val_preds, y_test)
          val_acc =    np.mean(np.argmax(y_test, axis=1) == val_preds.numpy())
          history['val_acc'].append(val_acc)
          if self.num_features == 28*28:
            print('Val acc:', val_acc, 'Val loss:', val_loss)
          else:
            f1_score_val = f1_score(val_preds, y_test_aux)
            history['val_f1'].append(f1_score_val)
            print('Val_f1:',f1_score_val,'Val acc:', val_acc)
          if flag2 == 0:
            flag2 = 1
            if  trigger<0.1: #(np.abs(history['train_loss'][-2]-history['train_loss'][-1]/history['train_loss'][-2])>=trigger):
              print('AG trigger') 
              # GA Parameters
              num_max_units = 128
              num_min_units = 10
              num_max_layers = 5
              num_min_layers = 2
              input_dim = self.num_features
              output_dim = self.num_classes
              n_stall_generations = 10
              n_iters = 20
              sample_update = 0.1
              # Invoke GA class 
              net_algo = self.copy_actual_instance()
              algo = GA(net_algo, sample_update, num_max_units, num_min_units, num_max_layers, num_min_layers, input_dim, output_dim, datagen)
              # Run GA optimization
              algo.run_algo(n_stall_generations, n_iters)
              # Save best NN & update params
              best_NN = algo.best_NN
              self.layers = best_NN.layers
              prev_L = self.L
              self.L = len(self.layers) # input, hidden & output layer
              for i in range(1, np.max(self.L, prev_L)):
                  if i <=(np.min(self.L, prev_L)):
                    self.W[i].assign(best_NN.W[i])
                    self.b[i].assign(best_NN.b[i])
                    if self.adam:
                        self.m_dW[i].assign(best_NN.m_dW[i])
                        self.v_dW[i].assign(best_NN.v_dW[i])
                        self.m_db[i].assign(best_NN.m_db[i])
                        self.v_db[i].assign(best_NN.v_db[i])
                  else:
                      if self.L > prev_L:
                        self.W[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) # to enable future modifications
                        self.b[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
                        self.W[i].assign(best_NN.W[i])
                        self.b[i].assign(best_NN.b[i])
                        if self.adam == 1:
                          self.m_dW[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) # to enable future modifications
                          self.m_db[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
                          self.m_dW[i].assign(best_NN.m_dW[i])
                          self.m_db[i].assign(best_NN.m_db[i])
                          self.v_dW[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) # to enable future modifications
                          self.v_db[i] = tf.Variable(1.0, shape=tf.TensorShape(None)) 
                          self.v_dW[i].assign(best_NN.v_dW[i])
                          self.v_db[i].assign(best_NN.v_db[i])
                      elif self.L < prev_L:
                         self.update_keys()
                         break
      return history
#-------------------------------------------------------------------------------
  def copy_actual_instance(self):
    copynet = FluidNetwork(self.layers)
    for i in range(1, copynet.L):
      copynet.W[i].assign(self.W[i])
      copynet.b[i].assign(self.b[i])
      if self.adam:
        copynet.m_dW[i].assign(self.m_dW[i])
        copynet.v_dW[i].assign(self.v_dW[i])
        copynet.m_db[i].assign(self.m_db[i])
        copynet.v_db[i].assign(self.v_db[i])
        copynet.t = 0 + 1 
    return copynet
#-------------------------------------------------------------------------------


class GA():
  "Each cromosome represents a possible structure of a new NN"
#-------------------------------------------------------------------------------
  def __init__(self, main_fluid_net, sample_params, num_max_units, num_min_units, num_max_layers, num_min_layers,input_dim, output_dim, datagen):
    self.num_max_units = num_max_units
    self.num_min_units = num_min_units
    self.num_max_layers = num_max_layers
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.pop_layers = []
    self.n_pop = []
    self.best_fitness = []
    self.best_NN = []
    self.best_layers = []
    self.pop_net = [] # the diff among pop_layers and pop_net is that the last one also incluse opt weights
    self.pop_fitness = []
    self.num_min_layers = num_min_layers
    self.pop_initial()
    self.main_fluid_net = main_fluid_net
    self.sample_params = sample_params # % of params to be sample out of a predetermined distribution
    self.pop_layers_prev = []
    self.actual_net = []
    self.datagen = datagen
#-------------------------------------------------------------------------------    
  def pop_initial(self):
    n_pop = int(np.ceil((self.input_dim * 50) / (5 + self.input_dim)))
    # n_pop = 10
    n_layers = np.random.randint(low=self.num_min_layers, high=self.num_max_layers, size=n_pop)
    
    pop_layers = [] # Pop of differents NN
    for i in range(n_pop):
      num_units_layer = np.random.randint(low=self.num_min_units, high=self.num_max_units, size=n_layers[i])
      pop_layers.append(num_units_layer)

    self.pop_initial = pop_layers
    self.n_pop = len(self.pop_initial)
    self.pop_layers = pop_layers

    # return pop_layers
#-------------------------------------------------------------------------------
  def fitness(self, epochs):
    # Reinitialize lists
    pop_fitness = []
    pop_net = []
    sample_update = np.round(self.n_pop/np.round(self.sample_params*self.n_pop, 0))
    j = 0 # to control sample udpate
    for i in range(np.max([self.n_pop, len(self.pop_layers)])):
      layers = self.pop_layers[i]
      if layers[0] != self.input_dim:
        layers = np.insert(layers, 0, self.input_dim)
      if layers[-1] != self.output_dim:
        layers = np.insert(layers, len(layers), self.output_dim)
      # Create de NN, starting from the previous step
      net = self.main_fluid_net.copy_actual_instance() # PUNTEROS!!
      # net = copy.deepcopy(self.main_fluid_net)
      net.new_topology = layers
      # Padding
      if j == sample_update:
        net.AG_update(0)
        j = 0 # start again
      else:
        net.AG_update(0) # 1: Sample distribution, 0: Normal initialization
      # Load data
      (x_train, y_train), (x_test, y_test) = load_data_mnist()
      # Training parameters
      batch_size = 128
      steps_per_epoch = int(x_train.shape[0]/batch_size)
      lr = 1e-2
      trigger = np.inf # to avoid an AG's trigger action
      # Training loop
      print(f'NN structure {i}/{self.n_pop}')
      # self.actual_net = net
      history = net.train(x_train, y_train, x_test, y_test, epochs, self.datagen,
        batch_size, lr, trigger)
      pop_fitness.append(history['val_acc'][-1])
      pop_net.append(net)
      # self.pop_layers.append(layers)
      j +=1
    
    # self.pop_layers = pop_layers
    self.pop_fitness = pop_fitness
    self.pop_net = pop_net 
    return pop_fitness
#-------------------------------------------------------------------------------
  def selection(self):
    pop_fitness =  self.pop_fitness
    
    # Although the fitness values (accuracy) are normalised, as they are compared 
    # to a random threshold between 0-1 we rather apply a min-max scaler.
    fitness_pu = np.asarray(pop_fitness)
    fitness_pu = (fitness_pu - fitness_pu.min()) / (fitness_pu.max() - fitness_pu.min()) 
    index = random.sample(range(len(fitness_pu)), len(fitness_pu))

    fathers_to_select = []
    j = 0
    while (len(fathers_to_select) < 2 and j < len(index)):
      i = index[j]
      threshold = np.random.random()
      if fitness_pu[i] > threshold:
        fathers_to_select.append(i)
      j +=1
    return fathers_to_select
#-------------------------------------------------------------------------------
  def crossover(self, crossover_rate, nchild): # Two children by each recombination
    threshold = np.random.random()
    if crossover_rate > threshold: # Then crossover operator is computed
      # Select crossover point that is not on the end of the string
      c = self.permutation2(2) # number of children
      # Wrap up all children
      # c1, c2 = list(c.values())
      return c
    else:
      return []
#-------------------------------------------------------------------------------
  def permutation(self, nchild):
    c = {}
    fathers_to_select = self.selection()
    if len(fathers_to_select) < 2:
      return []
    else:
      p1 = self.pop_layers_prev[fathers_to_select[0]]
      p2 = self.pop_layers_prev[fathers_to_select[1]]
      w_dW1 = np.abs(self.compute_db_weight(fathers_to_select[0]))
      w_dW2 = np.abs(self.compute_db_weight(fathers_to_select[1]))
      for j in range(2):
        c[j] = []
        if np.max([len(p1), len(p2)]) == self.num_min_layers:
              n = self.num_min_layers
        else:
          n = np.random.randint(self.num_min_layers, (np.max([len(p1), len(p2)]))) # child's length
        for i in range(n):
          if i < min([len(p1)-1, len(p2)-1]):
            idx = np.random.randint(1, np.min([len(p1), len(p2)])) - 1# -1 OJO
            aux = np.argmin([w_dW1[idx], w_dW2[idx]])
            if aux == 0:
              c[j].append(p1[idx])
            else:
              c[j].append(p2[idx])
          else:
            if len(p1) > len(p2):
              idx = np.random.randint(1, len(p1))
              c[j].append(p1[idx])
            else:
              idx = np.random.randint(1, len(p2))
              c[j].append(p2[idx])
      return c
#-------------------------------------------------------------------------------
  def permutation2(self, nchild):
    c = {}
    fathers_to_select = self.selection()
    if len(fathers_to_select) < 2:
      return []
    else:
      p1 = self.pop_layers_prev[fathers_to_select[0]]
      p2 = self.pop_layers_prev[fathers_to_select[1]]
      w_dW1 = np.abs(self.compute_db_weight(fathers_to_select[0]))
      w_dW2 = np.abs(self.compute_db_weight(fathers_to_select[1]))
      for j in range(2):
        c[j] = []
        if np.max([len(p1), len(p2)]) == self.num_min_layers:
              n = self.num_min_layers
        else:
          n = np.random.randint(self.num_min_layers, (np.max([len(p1), len(p2)]))) # child's length
        for i in range(n):
          if i < min([len(p1)-1, len(p2)-1]):
            idx = i
            aux = np.argmin([w_dW1[idx], w_dW2[idx]])
            if aux == 0:
              c[j].append(p1[idx])
            else:
              c[j].append(p2[idx])
          else:
            if len(p1) > len(p2):
              idx = i
              c[j].append(p1[idx])
            else:
              idx = i
              c[j].append(p2[idx])
      return c
#-------------------------------------------------------------------------------
  def mutation(self, mutation_rate):
    threshold = np.random.random()
    if threshold < mutation_rate:
       # n_mut_pop = np.int(self.n_pop*mutation_rate)
       n_mut_pop = 1
       return np.random.randint(low=self.num_min_layers, high=self.num_max_layers, size=n_mut_pop)
    else:
       return []
#-------------------------------------------------------------------------------
  def generational_replacement(self): 
    next_pop = []
    self.pop_layers_prev = self.pop_layers
    self.pop_layers = []
    i = 0
    next_pop.append(self.best_layers)
    while len(next_pop) < self.n_pop :
      aux1 = self.crossover(0.8, 2)
      aux2 = self.mutation(0.2)
      if len(aux1) != 0:
        aux1_0 = list(aux1.values())[0]
        aux1_1 = list(aux1.values())[1]
        next_pop.append(aux1_0)
        next_pop.append(aux1_1)
      if len(aux2) != 0:
        next_pop.append(list(aux2))
      i +=1
    self.pop_layers = next_pop
#-------------------------------------------------------------------------------
  def compute_db_weight(self, j):
    mean_dW = []
    net = self.pop_net[j]
    fitness = self.pop_fitness[j]
    for i in range(1, net.L):
      mean_dW.append(np.mean(net.dW[i]))
    return mean_dW/fitness
#-------------------------------------------------------------------------------
  def run_algo(self, n_stall_generations, max_iters):
     # Parameters' initialization
     self.best_fitness = 0
     stall_generations = 0
     iters = 0
     # While-loop to find the optimal parameters
     while stall_generations < n_stall_generations or iters < max_iters:
       if iters >= max_iters:
           break
       iters += 1
       pop_fitness = self.fitness(10) # numero de entrenamiento gradual con la iteracion, a mayor iteracion mayor epocas
       if self.best_fitness < max(pop_fitness):
         self.best_fitness = max(pop_fitness)
         max_index = pop_fitness.index(self.best_fitness)
         self.best_NN = self.pop_net[max_index]
         self.best_layers = self.pop_layers[max_index]
         stall_generations = 0
         print(f'Iter {iters}.The best NN has an accuracy of {self.best_fitness }')
       else:
         stall_generations += 1
         print(f'Iter {iters}.No improvement')
      
       self.generational_replacement()
     return self.best_NN
      