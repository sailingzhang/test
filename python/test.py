import os
import sys

work_dir = os.environ.get("WORK_DIR")
if work_dir is None:
    print("work_dir environment is None")
    sys.exit()
print("work_dir={}".format(work_dir))
sys.path.append(work_dir)
sys.path.append(work_dir+"\\test\\python")

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2,DQN
from gym_trading.envs.forex_env import forex_candle_env,ValidationRun


import logging
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append("..")
from log_init import log_init




import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
import torch.autograd.variable as variable



def filterTest():
    onlist=[x for x in range(10)]
    newlist =list(filter(lambda x:(x%2==0),onlist))
    logging.debug("newlist={}".format(newlist))
    newlist = list(map(lambda x:x*2,onlist))
    logging.debug("newlist={}".format(newlist))

def myyield(n):
    i =n
    while i>0:
        yield i
        i-=1

def yieldtest():
    f = myyield(2)
    for i in f:
        logging.debug("i={}".format(i))
    for j in myyield(3):
        logging.debug("j={}".format(j))


def gatherTest():
    t1 = torch.Tensor([[1,2,3],[4,5,6]])
    logging.debug("t1.size={}".format(t1.size()))
    logging.debug("t1={}".format(t1))
    newt1 = torch.gather(t1,dim=1,index=torch.LongTensor([[0,1],[2,0]]))
    logging.debug("newt1={}".format(newt1))

    newt2 =torch.gather(t1,dim=0,index=torch.LongTensor([[0,1,1],[0,0,0]]))
    logging.debug("newt2={}".format(newt2))


def squeezeTest():
    sq1 =torch.Tensor([[[1,2,3]],[[4,5,6]]])
    logging.debug("sq1'size={}".format(sq1.size()))
    asq2 = sq1.squeeze(1)
    logging.debug("asq2={}".format(asq2))
    asq3 = asq2.unsqueeze(2)
    logging.debug("asq3={}".format(asq3))
    
def maxTest():
    one = torch.Tensor([[2,3,1,5],[3,3,5,1]])
    one_0 =one.max(0)
    one_1 = one.max(1)
    logging.debug("one_0={},one_1={}".format(one_0,one_1))



def tenfortest():
    test=torch.Tensor([[1,2,3],[4,5,6]])
    testdata = test.data
    logging.debug("type(test)={},type(testdata)={},id(test)={},id(testdata)={}".format(type(test),type(testdata),id(test),id(testdata)))
    logging.debug("testdata={}".format(testdata))


class Base(object):
    def __init__(self,input):
        logging.debug("input={}".format(input))
    
class Child(Base):
    def fun1(self):
        logging.debug("hi hello")

def classTest():
    a = Child(2)
    a.fun1()




class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()  #b,3,32,32
        layer1=nn.Sequential()
        layer1.add_module('conv1_zhang',nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1))
        #b,32,32,32
        layer1.add_module('relu1',nn.ReLU(True))
        layer1.add_module('pool1',nn.MaxPool2d(2,2))
        #b,32,16,16
        self.layer1=layer1
        layer2=nn.Sequential()
        layer1.add_module('conv2',nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1))
        #b,64,16,16
        layer2.add_module('relu2',nn.ReLU(True))
        layer2.add_module('pool2',nn.MaxPool2d(2,2))
        #b,64,8,8
        self.layer2=layer2

        layer3=nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3 ,stride=1, padding = 1)) 
        #b,128,8,8
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('poo13', nn.MaxPool2d(2, 2))#b,128,4,4
        self.layer3=layer3

        layer4 =nn.Sequential()
        layer4.add_module('fc1',nn.Linear(in_features=2048, out_features=512 ))
        layer4.add_module('fc_relu1', nn.ReLU(True))
        layer4.add_module('fc2 ', nn.Linear(in_features=512, out_features=64 ))
        layer4.add_module('fc_relu2', nn.ReLU(True))
        layer4.add_module('fc3', nn.Linear(64, 10))
        self.layer4 = layer4

    def forward(self,x):
        conv1=self.layer1(x)
        conv2=self.layer2(conv1)
        conv3=self.layer3(conv2)
        fc_input=conv3.view(conv3.size(0),-1)
        fc_output=self.layer4(fc_input)
        return fc_output


def biasTest():
    model=SimpleCNN()
    for param in model.named_parameters():
        logging.debug("parar[0]={},parame[1].data.size()={}".format(param[0],param[1].data.size()))




from itertools import combinations
def calSet():
    R = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n' ])
    C = set(['b', 'd', 'f', 'l', 'n'])
    RR=[]
    RR.append(set([ 'a', 'c', 'e', 'g', 'i', 'k', 'l', 'm']))
    RR.append(set([ 'b', 'c', 'd', 'h', 'k']))
    RR.append(set([ 'd', 'f', 'g', 'n']))
    RR.append(set([ 'b', 'f', 'g', 'i', 'j']))
    RR.append(set([ 'b', 'k', 'n']))
    logging.debug("RR={}".format(RR))
    oriData={0,1,2,3,4}
    for i in range(1,6):
        for combina in combinations(oriData, i):
            union=set()
            for index in combina:
                union = union | RR[index]
            logging.debug("i={},combina={},union={}".format(i,combina,union))
            if union & C == C:
                logging.info("anser: combina={},len(combina)={},union={}".format(combina,len(combina),union))
                # return
#############################################stable baseline test begin#####################################################################
def baseLineLoadTest():
    FOREX_DATA_PATH=work_dir+ "/gym_trading/data/FOREX_EURUSD_1H_ASK_CLOSE.csv"
    env = forex_candle_env(FOREX_DATA_PATH, window_size=600,initCapitalPoint=2000,feePoint=20)
    env = DummyVecEnv([lambda: env])
    model = PPO2(MlpPolicy, env, verbose=1)
    # model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=20000)
    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        logging.debug("info={}".format(info))
        # env.render() 
#############################################stable baseline test end#####################################################################

if __name__ == "__main__":
    log_init("test.log","DEBUG")
    # filterTest()
    # yieldtest()
    # gatherTest()
    # squeezeTest()
    # maxTest()
    # tenfortest()
    # classTest()
    # biasTest()
    # calSet()
    baseLineLoadTest()
