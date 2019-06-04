
from log_init import log_init
import logging
import gym
import random
import torch
import torch.nn as nn
from data import mnist_data




class MyModule(nn.Module):
    def __init__(self,num_inputs,num_classes,droup_prob = 0.3):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs,6),
            nn.ReLU(),
            nn.Linear(6,num_classes),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
    def predict(self, x):
        return self.pipe(x)

    def fit(self,X,y):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)
        for i in range(1000):
            y_pred = self.pipe(X)
            logging.debug("y_pred={}".format(y_pred))
            loss = self.criterion(y_pred,y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    

def mymoduleTest():
    mindata = mnist_data()
    net = MyModule(num_inputs=2,num_classes=5)
    net.fit(mindata.X_train,mindata.y_train)
    # out = net.pipe(v)
    # logging.debug("out={}".format(out))        



if __name__ == '__main__':
    log_init("pytorch.log")
    mymoduleTest()
