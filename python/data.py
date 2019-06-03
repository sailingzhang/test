import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append("..")

# import tensorflow as tf
# import pandas as pd
# import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import fetch_mldata
# from datetime import datetime
import numpy as np
import mnist_diy
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class mnist_data:
    def __init__(self):
        traindata = list(mnist_diy.read(dataset="training",path="../notebookdata"))
        testdata =list(mnist_diy.read(dataset="testing",path="../notebookdata"))
        y_train =[x[0] for x in traindata]
        X_train =[x[1] for x in traindata]
        y_test =[x[0] for x in testdata]
        X_test =[x[1] for x in testdata]
        self.X_train = np.array(X_train).reshape(len(traindata),28*28).astype(np.float32)
        self.X_train_bais =np.c_[np.ones((len(self.X_train),1)),self.X_train].astype(np.float32)
        self.y_train =np.array(y_train).reshape(len(y_train),1).astype(np.int32)
        self.X_test =np.array(X_test).reshape(len(X_test),28*28).astype(np.float32)
        self.X_test_bais =np.c_[np.ones((len(self.X_test),1)),self.X_test].astype(np.float32)
        self.y_test =np.array(y_test).reshape(len(y_test),1).astype(np.int32)
        self.X_train_bais_scaled =StandardScaler().fit_transform(self.X_train_bais)
        self.X_test_bais_scaled = StandardScaler().fit_transform(self.X_test_bais)
        self.X_train_scaled = StandardScaler().fit_transform(self.X_train)
        self.X_test_scaled =  StandardScaler().fit_transform(self.X_test)
        print("type=",self.y_train.dtype)