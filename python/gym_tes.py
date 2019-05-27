
from log_init import log_init
import logging
import gym


def gym_test1():
    env = gym.make('CartPole-v0')
    env.reset()
    for i in range(1000):
        env.render()
        stepRsp=env.step(env.action_space.sample()) # take a random action
        logging.debug("i={},setpRsp={}".format(i,stepRsp))
    env.close()


class wa


if __name__ == '__main__':
    log_init("gymtest.log")
    gym_test1()
