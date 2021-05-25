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

  def contrastive_divergence(self, lr, k, input):
    self.input = input
    
    ph_mean, ph_sample = self.sample_h_given_v(self.input)
    start_sample = ph_sample

    for step in range(k):
        if step == 0:
            nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(start_sample)
        else:
            nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples)

    # Update parameters
    self.W += lr * (np.dot(self.input.T, ph_sample) - np.dot(nv_samples.T, nh_means))
    self.vbias += lr * np.mean(self.input - nv_samples, axis=0)
    self.hbias += lr * np.mean(ph_sample - nh_means, axis=0)

#=================================================================================================================#
#To obtain h sample from v sample

  def sample_h_given_v(self, v0_sample):
    pre_act = np.dot(v0_sample, self.W) + self.hbias
    h1_mean = self.sigmoid(pre_act)
    h1_sample = self.np_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean)

    return h1_mean, h1_sample

#=================================================================================================================#
#To obtain v sample from h sample

  def sample_v_given_h(self, h0_sample):
    pre_act = np.dot(h0_sample, self.W.T) + self.vbias
    v1_mean = self.sigmoid(pre_act)
    v1_sample = self.np_rng.binomial(size=v1_mean.shape, n=1, p=v1_mean)
    
    return v1_mean, v1_sample
  
#=================================================================================================================#
#to obtain new h sample from given h sample

  def gibbs_hvh(self, h0_sample):
    v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
    h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

    return v1_mean, v1_sample, h1_mean, h1_sample

#=================================================================================================================#

  def sigmoid(self, x):
    return 1. / (1 + np.exp(-x))

#=================================================================================================================#
#To get training RBM loss

  def get_reconstruction_loss(self, input):
    self.input=input
    pre_act_h = np.dot(self.input, self.W) + self.hbias
    sigmoid_act_h = self.sigmoid(pre_act_h)
    
    pre_act_v = np.dot(sigmoid_act_h, self.W.T) + self.vbias
    sigmoid_act_v = self.sigmoid(pre_act_v)

    ce_loss =  - np.mean(np.sum(self.input * np.log(sigmoid_act_v) + (1 - self.input) * np.log(1 - sigmoid_act_v), axis=1))
    
    return ce_loss

#=================================================================================================================#
#To obtain recontructed images using learned weights of rbm

  def get_reconstructed_image(self, v):
    h = self.sigmoid(np.dot(v, self.W) + self.hbias)
    reconstructed_v = self.sigmoid(np.dot(h, self.W.T) + self.vbias)
    return h, reconstructed_v

#=================================================================================================================#
#Creating Object 
rbm = RBM(visible_dim, hidden_dim)
#=================================================================================================================#
# main algorithm starts here

for epoch in range(max_epochs):
  for i in range(example):
    if method_of_rbm == 'Gibbs sampling':
      rbm.Block_Gibbs_sampling(lr=lr, k=k, r = r,input=trainData[i:i+1])
    else:
      rbm.contrastive_divergence(lr=lr, k=steps_Gibbs, input=trainData[i:i+1])
  training_loss = rbm.get_reconstruction_loss(input=trainData[0:example])
  print('Training epoch %d' % epoch)
  print('Training Loss: ',training_loss)
  
  train_hidden_rep, reconstructed_train = rbm.get_reconstructed_image(trainData[0:example])
  valid_hidden_rep, reconstructed_valid = rbm.get_reconstructed_image(validData[0:valid_example])
  test_hidden_rep, reconstructed_test = rbm.get_reconstructed_image(testData[0:test_example])

  logistic_regression= LogisticRegression(max_iter=2000)
  logistic_regression.fit(valid_hidden_rep,validLabel[0:valid_example])
  y_pred_probs = logistic_regression.predict_proba(test_hidden_rep)
  y_pred = logistic_regression.predict(test_hidden_rep)
  Test_Accuracy = metrics.accuracy_score(y_pred, testLabel[:test_example])*100
  Test_Loss = metrics.log_loss(testLabel[:test_example], y_pred_probs)
  print("Accuracy: ",metrics.accuracy_score(y_pred, testLabel[:test_example])*100)
  print("Test Loss: ",metrics.log_loss(testLabel[:test_example], y_pred_probs))
  print("-------------------------------------------------------")
  del y_pred,y_pred_probs,logistic_regression, train_hidden_rep, reconstructed_train, valid_hidden_rep, reconstructed_valid, test_hidden_rep, reconstructed_test
