
from log_init import log_init
import logging
import gym
import random


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





if __name__ == '__main__':
    log_init("gymtest.log")
    # gym_test1()
    mywrapperTest()
