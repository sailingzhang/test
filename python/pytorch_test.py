
from log_init import log_init
import logging
import gym
import random
import torch
import torch.nn as nn
import copy
from data import mnist_data
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
import pickle




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


saveParameter = "mnist_gen.data"
saveStad ="mnist_std.data"
class MyGen_net(nn.Module):
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
    def load(self,parameterfile=None,stdfile=None):
        if stdfile is not None:
            with open(stdfile,"rb+") as f:
                self.standardScaler_fit = pickle.load(f)
        if parameterfile is not None:
            self.load_state_dict(torch.load(parameterfile))
    def save(self,parameterfile,stdfile,):
        torch.save(self.state_dict(),parameterfile)
        with open(stdfile, 'wb') as f:
            pickle.dump(self.standardScaler_fit,f)

    def set_standardScaler_fit(self,standardScaler_fit):
        self.standardScaler_fit = standardScaler_fit
    def predict(self, X):
        X_scaled = self.standardScaler_fit.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        logists = self.pipe(X_tensor)
        proability_tensor =self.sm(logists)
        # logging.debug("proability_tensor={}".format(proability_tensor))
        proability = proability_tensor.detach().numpy()
        y_pred =np.argmax(proability,axis=1)
        return y_pred
    def evaluate(self,X,y):
        y_pred =self.predict(X)
        score = accuracy_score(y,y_pred)
        # logging.debug("evluate={}".format(score))
        return score


   

class trainClassifyGenNet():
    def __init__(self,net,populationNum,parentsNum,noiseStd,saveParameterFile,saveStdFile):
        self.net = net
        self.nets =[]
        self.populationNum = populationNum
        self.parentsNum = parentsNum
        self.noiseStd = noiseStd
        self.saveParameterFile = saveParameterFile
        self.saveStdFile = saveStdFile
        for _ in range(self.populationNum):
            self.nets.append(copy.deepcopy(net))

    def mutate_parent(self,net):
        new_net = copy.deepcopy(net)
        for p in new_net.parameters():
            noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
            p.data += self.noiseStd * noise_t
        return new_net

    def fit(self,X,y):
        writer = SummaryWriter(comment="-classify-gen")
        self.standardScaler_fit = StandardScaler().fit(X)
        for _,val in enumerate(self.nets):
            val.set_standardScaler_fit(self.standardScaler_fit)
        population = [ (net, net.evaluate(X,y)) for net in self.nets]
        # X_scaled = self.standardScaler_fit.transform(X)

        cur_max_rewards=0
        gen_idx =0
        while True:
            population.sort(key=lambda p: p[1], reverse=True)

            rewards = [p[1] for p in population[:self.parentsNum]]
            reward_mean = np.mean(rewards)
            reward_max = np.max(rewards)
            reward_std = np.std(rewards)
            writer.add_scalar("reward_mean", reward_mean, gen_idx)
            writer.add_scalar("reward_std", reward_std, gen_idx)
            writer.add_scalar("reward_max", reward_max, gen_idx)
            logging.debug("gen_idx={},reward_mean={},reward_std={},reward_max={}".format(gen_idx,reward_mean,reward_std,reward_max))
            if reward_max > cur_max_rewards:
                population[0][0].save(self.saveParameterFile,self.saveStdFile)
                if reward_max > 0.99:
                    return


            prev_population = population
            population = [population[0]]
            for _ in range(self.populationNum-1):
                parent_idx = np.random.randint(0, self.parentsNum)
                parent = prev_population[parent_idx][0]
                net = self.mutate_parent(parent)
                fitness = net.evaluate(X,y)
                population.append((net, fitness))
            gen_idx += 1            



def classifyGenTest():
    NOISE_STD = 0.01
    POPULATION_SIZE = 50
    PARENTS_COUNT = 10
    mindata = mnist_data()
    net = MyGen_net(num_inputs=28*28,num_classes=10)
    net.load(saveParameter,saveStad)
    trainInstance = trainClassifyGenNet(net,POPULATION_SIZE,PARENTS_COUNT,NOISE_STD,saveParameter,saveStad)
    trainInstance.fit(mindata.X_train,mindata.y_train)
    net.load(saveParameter,saveStad)
    y_pred =net.predict(mindata.X_test)
    logging.debug("y_pred={}".format(y_pred))
    logging.debug(".........accuracy={}".format(accuracy_score(mindata.y_test,y_pred)))




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

class RNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_layers):
        super(RNN,self).__init__()
        self.input_size = input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm = torch.nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
    
    def forward(self,input):
        # input应该为(batch_size,seq_len,input_szie)
        logging.debug("type(input)={},input.size()={},input.size(0)={},input.size(1)={}".format(type(input),input.size(),input.size(0),input.size(1)))
        # DEBUG type(input)=<class 'torch.Tensor'>,input.size()=torch.Size([2, 4, 12]),input.size(0)=2,input.size(1)=4
        self.hidden = self.initHidden(input.size(0))
        out,self.hidden = self.lstm(input,self.hidden)
        return out,self.hidden
    
    def initHidden(self,batch_size):
        if self.lstm.bidirectional:
            return (torch.rand(self.num_layers*2,batch_size,self.hidden_size),torch.rand(self.num_layers*2,batch_size,self.hidden_size))
        else:
            return (torch.rand(self.num_layers,batch_size,self.hidden_size),torch.rand(self.num_layers,batch_size,self.hidden_size))


def lstmTest():
    input_size = 12
    hidden_size = 10
    num_layers = 3
    batch_size = 2
    model = RNN(input_size,hidden_size,num_layers)
    # input (seq_len, batch, input_size) 包含特征的输入序列，如果设置了batch_first，则batch为第一维
    input = torch.rand(2,4,12)
    out,hidden = model(input)
    # logging.debug("type(out)={},type(hidden)={}".format(type(out),type(hidden)))
    # logging.debug("type(out)={},out.size={},type(hidden)={},hidden.size={}".format(type(out),out.size(),type(hidden),hidden.size()))
    logging.debug("out.size={},hidden[0].size={},hidden[1].size={}".format(out.size(),hidden[0].size(),hidden[1].size()))
    # DEBUG out.size=torch.Size([2, 4, 10]),hidden[0].size=torch.Size([3, 2, 10]),hidden[1].size=torch.Size([3, 2, 10])


if __name__ == '__main__':
    log_init("pytorch.log")
    classifyGenTest()
    # mymoduleTest()
    # lstmTest()
