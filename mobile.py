# Importing liabraries
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

threshhold = 0.844

# Importing data
dataset = pd.read_csv("mobile_cleaned-1549119762886.csv")
features = ["aperture", "price", "battery_capacity", "internal_memory", "stand_by_time", "screen_size", "processor_rank"]
X = dataset[features]
Y = dataset[["is_liked"]]

class Preprocessing:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None
        
    def binarize(self, X, n):
        Y=[]
        if type(X) is pd.core.frame.DataFrame:
            X = np.asarray(X)
        for x in X:
            if(x >=n):
                Y.append(1)
            else:
                Y.append(0)
        return np.asarray(Y)
        
    def scaling(self, t):
        self.X = StandardScaler().fit_transform(self.X)
        scaler = StandardScaler().fit(self.Y)
        self.Y = scaler.transform(self.Y)
        narmalized_threshold = (t - scaler.mean_)/scaler.var_
        self.Y = self.binarize(self.Y, narmalized_threshold[0])        
        
    def TTS(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size = 0.2, random_state = 512, stratify = self.Y)
        
        
preprocess = Preprocessing(X, Y)
preprocess.scaling(threshhold)
preprocess.TTS()

class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None
    def model(self, X):
        sigmoid = 1/(1+math.exp(-np.dot(self.w, X)))
        if(sigmoid >= self.b):
            return 1
        else:
            return 0        
        """if(np.dot(self.w, X) >= self.b):
            return 1
        else:
            return 0"""
    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.model(x))
        return Y
    
    def grad_w(self, X, Y, y_pred):
        return (y_pred-Y)*X
    
    def grad_b(self, Y, y_pred):
        return (y_pred-Y)
    
    def loss(self, y, y_pred):
        return (y-y_pred)**2
    
    def fit(self, X, Y, epochs, lr):
        self.w = np.ones(7)
        self.b = 0
        accuracy = {}
        max_acc = 0
        wt_matrix = []
        loss_matrix = []
        for i in range(epochs):
            loss = 0
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                for j in range(0, 7):
                    self.w[j] = self.w[j] - lr*self.grad_w(x[j], y, y_pred)
                self.b = self.b - lr*self.grad_b(y, y_pred)
                loss = loss + self.loss(y, y_pred)
            wt_matrix.append(self.w)
            loss_matrix.append(loss)
            accuracy[i] = accuracy_score(self.predict(X), Y)
            if (accuracy[i] > max_acc):
                max_acc = accuracy[i]
                chkptw = self.w
                chkptb = self.b
            self.w = chkptw
            self.b = chkptb 
        return loss_matrix
            
per = Perceptron()
loss = per.fit(preprocess.X_train, preprocess.y_train, 100, 0.24)
prediction2 = per.predict(preprocess.X_train)
print("Accuracy Score for train:"+str(accuracy_score(prediction2, preprocess.y_train))+"\n")
prediction1 = per.predict(preprocess.X_test)
print("Accuracy Score for test:"+str(accuracy_score(prediction1, preprocess.y_test))+"\n")