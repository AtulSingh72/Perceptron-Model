# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:08:49 2020

@author: Samarth Tibdewal
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
THRESHOLD=0.1

# Perceptron using loss function as Mean Square Error
class PERCEPTRONmse:
    def __init__(self, features, predictor):
        self.features=features
        self.predictor=predictor
    
    def Scaling(self, features):
        scaled_features=StandardScaler().fit_transform(features)
        return np.c_[scaled_features, np.ones([scaled_features.shape[0],1])]
    
    def Activation(self, hypothesis):
        activ=(1/(1+np.exp(-(hypothesis))))
        #print(activ)
        return activ        
        
    def Hypothesis (self, scaled_features,weights):
        hypo=np.dot(scaled_features,weights)
        output=self.Activation(hypo)
        return output
    
    def Binarize(self, prediction):
        if(prediction >= THRESHOLD):
            return 1
        else:
            return 0
    
    def Loss_Func(self,prediction, predictor):
        loss= np.square((prediction-predictor))
        return loss[0]
    
    def Fit(self,epochs,lr):
        self.scaled_features=self.Scaling(self.features)
        out_hist=[]
        predictions = []
        loss_hist=[]
        accuracy=[]
        max_accuracy=0
        loss_iter=100
        self.weights=np.ones([self.scaled_features.shape[1]])
        #print(self.weights.shape)
        while (loss_iter>=1):
            for i in range(epochs):
                for x,y in zip(self.scaled_features,self.predictor):
                    output=self.Hypothesis(x, self.weights)
                    out_hist.append(output)
                    loss_iter+=self.Loss_Func(output,y)
                    self.weights[0]=self.weights[0]-(lr*(output-y)*x[0])
                    self.weights[1]=self.weights[1]-(lr*(output-y)*x[1])
                    self.weights[2]=self.weights[2]-(lr*(output-y)*x[2])
                    self.weights[3]=self.weights[3]-(lr*(output-y)*x[3])
                    self.weights[4]=self.weights[4]-(lr*(output-y)*x[4])
                    self.weights[5]=self.weights[5]-(lr*(output-y)*x[5])
                    self.weights[6]=self.weights[6]-(lr*(output-y)*x[6])
                    predictions.append(self.Binarize(output))
                #predictions = self.Binarize(output)
                loss_hist.append(loss_iter)
                loss_iter=0
                accuracy.append(accuracy_score(predictions,self.predictor))
                if(accuracy[i]>max_accuracy):
                    max_accuracy=accuracy[i]
                    chkptw=self.weights
                out_hist.clear()
                predictions.clear()
            print(max_accuracy)
            return [chkptw, max_accuracy, loss_hist]

# Perceptron using loss function as Binary Cross Entropy
class PERCEPTRONbct:
    def __init__(self, features, predictor):
        self.features=features
        self.predictor=predictor
    
    def Scaling(self, features):
        scaled_features=StandardScaler().fit_transform(features)
        return np.c_[scaled_features, np.ones([scaled_features.shape[0],1])]
    
    def Activation(self, hypothesis):
        activ=(1/(1+np.exp(-(hypothesis))))
        #print(activ)
        return activ        
        
    def Hypothesis (self, scaled_features,weights):
        hypo=np.dot(scaled_features,weights)
        output=self.Activation(hypo)
        return output
    
    def Binarize(self, prediction):
        if(prediction >= THRESHOLD):
            return 1
        else:
            return 0
    
    def Loss_Func(self,prediction, predictor):
        #loss= np.square((prediction-predictor))
        loss = (1 - predictor)*np.log(1-prediction)+(predictor * np.log(prediction))
        #print(loss[0])
        return loss[0]
    
    def Fit(self,epochs,lr):
        self.scaled_features=self.Scaling(self.features)
        out_hist=[]
        predictions = []
        loss_hist=[]
        accuracy=[]
        max_accuracy=0
        loss_iter=100
        self.weights=np.ones([self.scaled_features.shape[1]])
        #print(self.weights.shape)
        while (loss_iter>=1):
            for i in range(epochs):
                for x,y in zip(self.scaled_features,self.predictor):
                    output=self.Hypothesis(x, self.weights)
                    out_hist.append(output)
                    loss_iter+=self.Loss_Func(output,y)
                    self.weights[0]=self.weights[0]-(lr*(output-y)*x[0])
                    self.weights[1]=self.weights[1]-(lr*(output-y)*x[1])
                    self.weights[2]=self.weights[2]-(lr*(output-y)*x[2])
                    self.weights[3]=self.weights[3]-(lr*(output-y)*x[3])
                    self.weights[4]=self.weights[4]-(lr*(output-y)*x[4])
                    self.weights[5]=self.weights[5]-(lr*(output-y)*x[5])
                    self.weights[6]=self.weights[6]-(lr*(output-y)*x[6])
                    predictions.append(self.Binarize(output))
                #predictions = self.Binarize(output)
                loss_hist.append(loss_iter)
                loss_iter=0
                accuracy.append(accuracy_score(predictions,self.predictor))
                if(accuracy[i]>max_accuracy):
                    max_accuracy=accuracy[i]
                    chkptw=self.weights
                out_hist.clear()
                predictions.clear()
            print(max_accuracy)
            return [chkptw, max_accuracy, loss_hist]

data=pd.read_csv("mobile_cleaned-1549119762886.csv")
data_thin=data[['aperture','battery_capacity','brand_rank','stand_by_time','screen_size','price','video_resolution']]
data_thin.head
predictor_target=data[['is_liked']].values

perc=PERCEPTRONbct(data_thin,predictor_target)
final=perc.Fit(2000,0.15)
wt_matrix = final[0]
loss_matrix = final[2]
plt.plot(range(0, 2000), final[2], '-')