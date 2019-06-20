
from log_init import log_init
import logging
import gym
import random
import torch
import torch.nn as nn
from data import mnist_data
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler




class MyModule(nn.Module):
    def __init__(self,num_inputs,num_classes,droup_prob = 0.3):
        super().__init__()
        # self.criterion = torch.nn.MSELoss(reduction='sum')
        self.sm = nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs,300),
            nn.ReLU(),
            nn.Linear(300,100),
            nn.ReLU(),
            nn.Linear(100,num_classes),
            # nn.ReLU(),
            # nn.Softmax(dim=1)
        )
    def predict(self, X):
        X_scaled = self.standardScaler_fit.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        logists = self.pipe(X_tensor)
        proability_tensor =self.sm(logists)
        # logging.debug("proability_tensor={}".format(proability_tensor))
        proability = proability_tensor.detach().numpy()
        y_pred =np.argmax(proability,axis=1)
        return y_pred

    def fit(self,X,y):
        logging.debug("X'shape={},y'shape={}".format(X.shape,y.shape))
        self.standardScaler_fit = StandardScaler().fit(X)
        X_scaled = self.standardScaler_fit.transform(X)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)
        # self.optimizer =torch.optim.Adam(params=self.parameters(), lr=0.05)
        for param in self.parameters():
            logging.debug("befor param={}".format(param))
        for i in range(200):
            logging.debug("i={}".format(i))
            rand_index = np.random.permutation(len(X_scaled))
            X_bachs = np.array_split(X_scaled[rand_index],10)
            y_bachs = np.array_split(y[rand_index],10)
            for xbach,ybach in  zip(X_bachs,y_bachs):
                X_tensor = torch.FloatTensor(xbach)
                y_tensor = torch.LongTensor(ybach)
                # logging.debug("type(X_tensor)={},type(y_tensor)={}".format(X_tensor,y_tensor))
                logists = self.pipe(X_tensor)
                # logging.debug("logists={}".format(logists))
                self.optimizer.zero_grad()
                loss = self.criterion(logists,y_tensor)
                # logging.debug("loss={}".format(loss))
                loss.backward()
                self.optimizer.step()
            logging.debug("logists'size={},y_tensor'size={}".format(logists.size(),y_tensor.size()))
            logging.debug("loss={}".format(loss))
            # for param in self.parameters():
            #     logging.debug("after param={}".format(param))
    

def mymoduleTest():
    mindata = mnist_data()
    net = MyModule(num_inputs=28*28,num_classes=10)
    # logging.debug("x_train={}".format(mindata.X_train[0]))
    logging.debug("y_train={}".format(mindata.y_train))
    net.fit(mindata.X_train,mindata.y_train)
    y_pred =net.predict(mindata.X_test)
    logging.debug("y_pred={}".format(y_pred))
    logging.debug("accuracy={}".format(accuracy_score(mindata.y_test,y_pred)))
    # out = net.pipe(v)
    # logging.debug("out={}".format(out))        



if __name__ == '__main__':
    log_init("pytorch.log")
    mymoduleTest()
