
from log_init import log_init
import logging
import gym
import random
import torch
import torch.nn as nn
from data import mnist_data
import numpy as np
from sklearn.metrics import accuracy_score




class MyModule(nn.Module):
    def __init__(self,num_inputs,num_classes,droup_prob = 0.3):
        super().__init__()
        # self.criterion = torch.nn.MSELoss(reduction='sum')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs,300),
            nn.ReLU(),
            nn.Linear(300,num_classes),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
    def predict(self, X):
        X_tensor = torch.FloatTensor(X)
        proability_tensor =self.pipe(X_tensor)
        # logging.debug("proability_tensor={}".format(proability_tensor))
        proability = proability_tensor.detach().numpy()
        y_pred =np.argmax(proability,axis=1)
        return y_pred

    def fit(self,X,y):
        logging.debug("X'shape={},y'shape={}".format(X.shape,y.shape))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)
        for param in self.parameters():
            logging.debug("befor param={}".format(param))
        for i in range(200):
            rand_index = np.random.permutation(len(X))
            X_bachs = np.array_split(X[rand_index],10)
            y_bachs = np.array_split(y[rand_index],10)
            for xbach,ybach in  zip(X_bachs,y_bachs):
                X_tensor = torch.FloatTensor(xbach)
                y_tensor = torch.LongTensor(ybach)
                # logging.debug("type(X_tensor)={},type(y_tensor)={}".format(X_tensor,y_tensor))
                y_pred = self.pipe(X_tensor)
                logging.debug("y_pred={}".format(y_pred))
                loss = self.criterion(y_pred,y_tensor)
                logging.debug("loss={}".format(loss))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            for param in self.parameters():
                logging.debug("after param={}".format(param))
    

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
