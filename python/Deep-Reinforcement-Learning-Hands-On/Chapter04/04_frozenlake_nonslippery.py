#!/usr/bin/env python3
import sys
sys.path.append("../../")
sys.path.append("../../ptan-master")
import logging
from log_init import log_init

import random
import gym
import gym.spaces
import gym.wrappers
import gym.envs.toy_text.frozen_lake
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    """
    返回的batch是一个生命周期，且是有序的。
    action 是通过net选取的。
    """
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])#[obs]是一个，不是batch
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))#batch里的每一个元素是一个episode的总reward和每step的(obs,action)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    """
    输入batch是历史batch
    """
    disc_rewards = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), batch))#为什么生命周期越长，reward会下降越快？可能是因为花的时间越短越好吧。
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:#这里进行筛选应该是保证收敛的前提。
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)

    return elite_batch, train_obs, train_act, reward_bound


"""
（1）收集episode
（2）根据每个episode的总rewards和step的长度对每个episode进行评分。
（3）选择一批最评分最高的episode,  best_episode_batch.
(4)把best_episode_batch的obs通过 net得到score.
(5)把best_episode_batch的action 当成目标分类。
(6)cross_entropy(score,action)

"""

if __name__ == "__main__":
    log_init("../../04_frozenlake_nonslippery.log")
    random.seed(12345)
    env = gym.envs.toy_text.frozen_lake.FrozenLakeEnv(is_slippery=False)
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=100)#env里有一个错误，暂时注释掉
    env = DiscreteOneHotWrapper(env)
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    logging.debug("env.observation_space.shape={}".format(env.observation_space.shape))
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)
    writer = SummaryWriter(comment="-frozenlake-nonslippery")

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):#返回的batch是一个生命周期，且是有序的。batch中只有all_rewards,obs,action，没有step reward, batch=[(),()]
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))#这里的reward是一个episode 的总reward.因为经过筛选，这里的reward_mean应该越来越高。
        full_batch, obs, acts, reward_bound = filter_batch(full_batch + batch, PERCENTILE)#叠加输入，返回被筛选后的原始batch,obs,acts,和筛选边界。full_batch=[(),()]
        if not full_batch:
            continue
        obs_v = torch.FloatTensor(obs)
        acts_v = torch.LongTensor(acts)
        full_batch = full_batch[-500:]

        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, batch=%d" % (
            iter_no, loss_v.item(), reward_mean, reward_bound, len(full_batch)))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        if reward_mean > 0.8:
            print("Solved!")
            break
    writer.close()
