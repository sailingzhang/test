#!/usr/bin/env python3
import sys
sys.path.append("../../")
sys.path.append("../../ptan-master")
import logging as log
from log_init import log_init

import gym
import pybullet_envs

ENV_ID = "MinitaurBulletEnv-v0"
RENDER = True


if __name__ == "__main__":
    log_init("../../chek_env.log")
    spec = gym.envs.registry.spec(ENV_ID)
    spec._kwargs['render'] = RENDER
    env = gym.make(ENV_ID)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print(env)
    print(env.reset())
    input("Press any key to exit\n")
    env.close()
