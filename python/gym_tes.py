
from log_init import log_init
import logging
import gym
import random
import torch
import torch.nn as nn


def gym_test1():
    env = gym.make('CartPole-v0')
    env.reset()
    for i in range(1000):
        env.render()
        stepRsp=env.step(env.action_space.sample()) # take a random action
        logging.debug("i={},setpRsp={}".format(i,stepRsp))
    env.close()


class RandomWrapper(gym.ActionWrapper):
    def __init__(self,env,espilon=0.5):
        super().__init__(env)
        self.espilon = espilon
    def action(self,action):
        if random.random() < self.espilon:
            logging.debug("Random")
            return self.env.action_space.sample()
        else:
            logging.debug("no Random")
            return action

def mywrapperTest():
    env = RandomWrapper(gym.make('CartPole-v0'))
    obs = env.reset()
    total_reward = 0.0
    while True:
        env.render()
        obs,reward,done,_ = env.step(0)
        if done:
            # pass
            break
    logging.debug("reward get={}".format(total_reward))


def torchGradientTest():
    v1 = torch.tensor([4.0,1.0],requires_grad = True)
    v2 = torch.tensor([2.0,2.0])
    v_sum = v1+v2
    v_res = (v_sum*2).sum()
    v_res.backward()
    logging.debug("v_res={},v1.grad={}".format(v_res,v1.grad))

def nntest():
    l = nn.Linear(2,5)
    v = torch.FloatTensor([1,2])
    logging.debug("l(v)={}".format(l(v)))
    s = nn.Sequential(
        nn.Linear(2,5),
        nn.ReLU()
    )

    logging.debug("s={}".format(s(v)))

class MyModule(nn.Module):
    def __init__(self,num_inputs,num_classes,droup_prob = 0.3):
        super().__init__()
        # super(MyModule,self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs,6),
            nn.ReLU(),
            nn.Linear(6,num_classes),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
    # def __call__(self,value_tenfor):   
    #     return self.pipe(value_tenfor)
    def forward(self, x):
        return self.pipe(x)

def mymoduleTest():
    net = MyModule(num_inputs=2,num_classes=5)
    v = torch.FloatTensor([[1,3]])
    out =net(v)
    # out = net.pipe(v)
    logging.debug("out={}".format(out))        

if __name__ == '__main__':
    log_init("gymtest.log")
    # gym_test1()
    # mywrapperTest()
    # torchGradientTest()
    # nntest()
    mymoduleTest()
