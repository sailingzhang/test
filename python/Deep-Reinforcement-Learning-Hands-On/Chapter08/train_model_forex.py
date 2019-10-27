#!/usr/bin/env python3
import sys
#sys.path.append("C:\\mydata\\develop\\mygit\\gym_trading")
sys.path.append("/home/sailingzhang/develop/mygit/gym_trading")
# sys.path.append(" ../../../../gym_trading")
sys.path.append("../../")
sys.path.append("../../ptan-master")

import logging
from log_init import log_init
logging.debug("load moundle")

import os
import gym
from gym import wrappers
import ptan
import argparse
import numpy as np

import torch
import torch.optim as optim

from lib import environ, data, models, common, validation

from tensorboardX import SummaryWriter
from envs.forex_env import forex_candle_env,ValidationRun


BATCH_SIZE = 32
BARS_COUNT = 10
TARGET_NET_SYNC = 1000
DEFAULT_STOCKS = "data/YNDX_160101_161231.csv"
DEFAULT_VAL_STOCKS = "data/YNDX_150101_151231.csv"

FOREC_DATA="C:\mydata\develop\mygit\gym_trading\data\FOREX_EURUSD_1H_ASK.csv"

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
#REPLAY_INITIAL = 1000

REWARD_STEPS = 2

LEARNING_RATE = 0.0001

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000

EPSILON_START = 1.0
EPSILON_STOP = 0.1
EPSILON_STEPS = 1000000

CHECKPOINT_EVERY_STEP = 1000000
VALIDATION_EVERY_STEP = 100000
#VALIDATION_EVERY_STEP = 1000

# FOREX_DATA_PATH="../../../../gym_trading/data/FOREX_EURUSD_1H_ASK.csv"
FOREX_DATA_PATH="../../../../gym_trading/data/FOREX_EURUSD_1H_ASK_CLOSE.csv"

def test2():
    logging.debug("enter")
    logging.info("info enter")
    device = torch.device("cuda")
    env = forex_candle_env(FOREX_DATA_PATH, window_size=600,initCapitalPoint=2000,feePoint=20)
    net = models.SimpleFFDQN_V(env.observation_space.shape[0], env.action_space.n).to(device)
    net.load_state_dict(torch.load("saves/forex/mean_val-1218668418.000.data"))
    ValidationRun(env,net,episodes= 5,device= device,epsilon= 0)
    logging.debug("exit")

def test():    
    print("fuck")
    logging.info("enter")
    # return
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--data", default=DEFAULT_STOCKS, help="Stocks file or dir to train on, default=" + DEFAULT_STOCKS)
    parser.add_argument("--year", type=int, help="Year to be used for training, if specified, overrides --data option")
    parser.add_argument("--valdata", default=DEFAULT_VAL_STOCKS, help="Stocks data for validation, default=" + DEFAULT_VAL_STOCKS)
    parser.add_argument("-r", "--run", required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    logging.info("device={}".format(device))

    saves_path = os.path.join("saves", args.run)
    os.makedirs(saves_path, exist_ok=True)



    # if args.year is not None or os.path.isfile(args.data):
    #     if args.year is not None:
    #         stock_data = data.load_year_data(args.year)
    #     else:
    #         stock_data = {"YNDX": data.load_relative(args.data)}#run here by default argument
    #     env = environ.StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False, volumes=False)
    #     env_tst = environ.StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)
    # elif os.path.isdir(args.data):
    #     env = environ.StocksEnv.from_dir(args.data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)
    #     env_tst = environ.StocksEnv.from_dir(args.data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)
    # else:
    #     raise RuntimeError("No data to train on")
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    # val_data = {"YNDX": data.load_relative(args.valdata)}
    # env_val = environ.StocksEnv(val_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)

    logging.debug("begin to load env")
    env = forex_candle_env(FOREX_DATA_PATH, window_size=600,initCapitalPoint=2000,feePoint=20)
    env_val = forex_candle_env(FOREX_DATA_PATH, window_size=600,initCapitalPoint=2000,feePoint=20)
    logging.info("env.observation_sapce={},env.action_space.n={}".format(env.observation_space,env.action_space.n))

    writer = SummaryWriter(comment="-simple-" + args.run)
    # net = models.SimpleFFDQN(env.observation_space.shape[0], env.action_space.n).to(device)
    net = models.SimpleFFDQN_V(env.observation_space.shape[0], env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # main training loop
    step_idx = 0
    eval_states = None
    best_mean_val = None
    logging.debug("begin train")
    with common.RewardTracker(writer, np.inf, group_rewards=100) as reward_tracker:
        while True:
            step_idx += 1
            if 0 == step_idx%1000:
                logging.info("step_idx={}".format(step_idx))
            buffer.populate(1)
            selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)

            new_rewards = exp_source.pop_rewards_steps()
            if new_rewards:
                reward_tracker.reward(new_rewards[0], step_idx, selector.epsilon)

            if len(buffer) < REPLAY_INITIAL:
                continue

            if eval_states is None:
                logging.debug("Initial buffer populated, start training")
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            if step_idx % EVAL_EVERY_STEP == 0:
                mean_val = common.calc_values_of_states(eval_states, net, device=device)
                writer.add_scalar("values_mean", mean_val, step_idx)
                if best_mean_val is None or best_mean_val < mean_val:
                    if best_mean_val is not None:
                        logging.debug("%d: Best mean value updated %.3f -> %.3f" % (step_idx, best_mean_val, mean_val))
                    best_mean_val = mean_val
                    torch.save(net.state_dict(), os.path.join(saves_path, "mean_val-%.3f.data" % mean_val))

            logging.debug("begin optimer,step_idx={}".format(step_idx))
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_v = common.calc_loss_V(batch, net, tgt_net.target_model, GAMMA ** REWARD_STEPS, device=device)
            loss_v.backward()
            optimizer.step()

            if step_idx % TARGET_NET_SYNC == 0:
                logging.info("begin sync,step_idex={},loss={}".format(step_idx,loss_v))
                tgt_net.sync()

            if step_idx % CHECKPOINT_EVERY_STEP == 0:
                logging.info("begin save,step_idx={}".format(step_idx))
                idx = step_idx // CHECKPOINT_EVERY_STEP
                torch.save(net.state_dict(), os.path.join(saves_path, "checkpoint-%3d.data" % idx))

            if step_idx % VALIDATION_EVERY_STEP == 0:
                logging.info("begin valid,step_idx={}".format(step_idx))
                ValidationRun(env_val,net,episodes= 1,device= device,epsilon= 0)
                # res = validation.validation_run(env_tst, net, device=device)
                # for key, val in res.items():
                #     writer.add_scalar(key + "_test", val, step_idx)
                # res = validation.validation_run(env_val, net, device=device)
                # for key, val in res.items():
                #     writer.add_scalar(key + "_val", val, step_idx)


if __name__ == "__main__":
    log_init("../../../../tmp/ch08_train.log",'DEBUG')
    test()
    # test2()
