import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow                     
from tensorflow import keras
from matplotlib import pyplot
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
#=================================================================================================================#
#setting hyperparameter values

method_of_rbm = 'Contrastive Divergence'   # put 'Gibbs sampling' for block sampling algorithm and 'Contrastive Divergence' for cd algorithm.
lr = 0.01
hidden_dim = 256        #[64, 128, 256]
steps_Gibbs = 10        #for contrastive divergence [1,5,10]
max_epochs = 1          # 100 for Contrastive Divergence and 10 for Block Gibbs Sampling
k = 200                 # markov chain runs for gibbs sampling [100, 200, 300]
r = 10                  # after convergence runs for gibbs sampling [10, 20, 30]
#=================================================================================================================#
#Fixed Parameters
visible_dim = 784
example = 42000        # 70% of 60k examples
valid_example = 18000  # 30% of 60k examples
test_example = 10000

# Load dataset
(trainData, trainLabel), (testData, testLabel) = mnist.load_data()

# Reshaping dataset
trainData = trainData.reshape(len(trainData),784)
trainData=(trainData>=127)
testData = testData.reshape(len(testData),784)
testData=(testData>=127)
trainData, validData, trainLabel, validLabel = train_test_split(trainData,trainLabel, test_size=0.3, shuffle=True)   
print("Dataset Loaded")

#=================================================================================================================#
class RBM(object):
  def __init__(self,  visible_dim, hidden_dim):
      
    self.visible_dim = visible_dim
    self.hidden_dim = hidden_dim

    self.np_rng = np.random.RandomState(0)

    # Initialize parameters
    width = 1. / visible_dim
    self.W = np.array(self.np_rng.uniform(low=-width, high=width, size=(visible_dim, hidden_dim)))

    self.hbias = np.zeros(hidden_dim)
    self.vbias = np.zeros(visible_dim)
    self.input = None

#=================================================================================================================#
  # To run the block sampling RBM

  def Block_Gibbs_sampling(self, lr, k, r, input):
    self.v_sample_list = []
    self.h_mean_list = []

    self.input = input
    self.input_r = np.random.choice([0, 1], size=(1,784), p=[1./2, 1./2])

    A = np.zeros((self.visible_dim,self.hidden_dim))
    B = np.zeros((1,self.visible_dim))
    C = np.zeros((1,self.hidden_dim))

    ph_mean, ph_sample = self.sample_h_given_v(self.input_r)
    initial_mean,initial_sample = self.sample_h_given_v(self.input)
    start_sample = ph_sample

    for step in range(k+r):
      if step == 0:
        nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(start_sample)
      else:
        nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples)
      self.v_sample_list.append(nv_samples)#cap_v_T
      self.h_mean_list.append(nh_means)

    for i in range(k,k+r):
      v_sample = self.v_sample_list[i]
      h_mean = self.h_mean_list[i]
      A += np.dot(v_sample.T, h_mean)
      B += v_sample
      C += h_mean

    # Update parameters
    self.W += lr * ((np.dot(self.input.T, initial_sample)) - (A/r))#ph_sample to ph_mean
    self.vbias += lr *np.mean(self.input - (B/r),axis=0)
    self.hbias += lr * np.mean(initial_sample - (C/r),axis=0)#ph_sample to ph_mean
    
#=================================================================================================================#  
  #To run the contrastive divergence RBM 
