#!/usr/bin/env python3


import os
import sys
# work_dir="C:\\mydata\\develop\\mygit"
# work_dir= "/home/sailingzhang/develop/mygit"
# work_dir ="/home/sailingzhang/winshare/develop/source/mygit"

work_dir = os.environ.get("WORK_DIR")
if work_dir is None:
    print("work_dir environment is None")
    sys.exit()
print("work_dir={}".format(work_dir))

sys.path.append(work_dir)
sys.path.append(work_dir+"\\test\\python")
sys.path.append("../../")
sys.path.append("../../ptan-master")

import logging
from log_init import log_init

logging.debug("load moundle")


import gym
from gym import wrappers
import ptan
import argparse
import numpy as np
import json
import time
import torch
import torch.optim as optim

from lib import environ, data, models, common, validation

from tensorboardX import SummaryWriter
from gym_trading.envs.forex_env import forex_candle_env,ValidationRun


# BATCH_SIZE = 32
BATCH_SIZE = 100
BARS_COUNT = 10
TARGET_NET_SYNC = 1000
DEFAULT_STOCKS = "data/YNDX_160101_161231.csv"
DEFAULT_VAL_STOCKS = "data/YNDX_150101_151231.csv"



FOREC_DATA="C:\mydata\develop\mygit\gym_trading\data\FOREX_EURUSD_1H_ASK.csv"

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
# REPLAY_INITIAL = 1000

REWARD_STEPS = 2

LEARNING_RATE = 0.0001

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000

EPSILON_START = 1.0
EPSILON_STOP = 0.1
EPSILON_STEPS = 1000000

# CHECKPOINT_EVERY_STEP = 1000000
CHECKPOINT_EVERY_STEP = 10000

# VALIDATION_EVERY_STEP = 100000
VALIDATION_EVERY_STEP = 10000

# FOREX_DATA_PATH="../../../../gym_trading/data/FOREX_EURUSD_1H_ASK.csv"
FOREX_DATA_PATH="../../../../gym_trading/data/FOREX_EURUSD_1H_ASK_CLOSE.csv"



P_TIMER_CHECK_DATA_FILE ="recent_train.data"
P_MAX_MEAN_DATA_FILE ="max_mean.data"
P_MAC_VALIDATE_DATA_FILE="max_validate.date"
P_DATA_INFO_FILE ="data_info.json"
INIT_PARAMETER = False

init_funcs = {
    1: lambda x: torch.nn.init.normal_(x, mean=0., std=1.), # can be bias
    2: lambda x: torch.nn.init.xavier_normal_(x, gain=1.), # can be weight
    3: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), # can be conv1D filter
    4: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), # can be conv2D filter
    "default": lambda x: torch.nn.init.constant(x, 1.), # everything else
}
def init_all(model, init_funcs = init_funcs):
    for p in model.parameters():
        init_func = init_funcs.get(len(p.shape), init_funcs["default"])
        init_func(p)

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
    data_info_dict={}
    # return
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--data", default=DEFAULT_STOCKS, help="Stocks file or dir to train on, default=" + DEFAULT_STOCKS)
    parser.add_argument("--year", type=int, help="Year to be used for training, if specified, overrides --data option")
    parser.add_argument("--valdata", default=DEFAULT_VAL_STOCKS, help="Stocks data for validation, default=" + DEFAULT_VAL_STOCKS)
    parser.add_argument("-r", "--run", required=True, help="Run name")
    parser.add_argument("--pretype",required=True)
    parser.add_argument("--init",default=False,help="just init")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    INIT_PARAMETER = args.init
    logging.info("device={}".format(device))

    saves_path = os.path.join("saves", args.run)
    os.makedirs(saves_path, exist_ok=True)







    logging.debug("begin to load env")
    env = forex_candle_env(FOREX_DATA_PATH, window_size=600,initCapitalPoint=2000,feePoint=20,preprocessType=args.pretype)
    env_val = forex_candle_env(FOREX_DATA_PATH, window_size=600,initCapitalPoint=2000,feePoint=20,preprocessType=args.pretype)
    logging.info("env.observation_sapce={},env.action_space.n={}".format(env.observation_space,env.action_space.n))

    writer = SummaryWriter(comment="-simple-" + args.run)
    # net = models.SimpleFFDQN(env.observation_space.shape[0], env.action_space.n).to(device)
    net = models.SimpleFFDQN_V(env.observation_space.shape[0], env.action_space.n).to(device)
    check_data_file = os.path.join(saves_path,P_TIMER_CHECK_DATA_FILE)
    if INIT_PARAMETER:
        # init_all(net)
        torch.save(net.state_dict(), check_data_file)
        logging.info("save,file={}".format(check_data_file))
        sys.exit()

    if False == os.path.exists(check_data_file):
        logging.error("load data file not exit:{}".format(check_data_file))
        return
    net.load_state_dict(torch.load(check_data_file))



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

    timeCount={"t1t2_sum":0,"t1t2_num":0,"t2t3_sum":0,"t2t3_num":0}
    with common.RewardTracker(writer, np.inf, group_rewards=100) as reward_tracker:
        while True:
            step_idx += 1
            t1 = time.time()
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
                    torch.save(net.state_dict(), os.path.join(saves_path, P_MAX_MEAN_DATA_FILE))
                    data_info_dict["best_mean_val"]=best_mean_val
                    with open(os.path.join(saves_path,P_DATA_INFO_FILE),"w") as f:
                        json.dump(data_info_dict,f)
                    logging.info("best_mean_val={}".format(best_mean_val))
            t2 = time.time()
            timeCount["t1t2_sum"] += (t2-t1)
            timeCount["t1t2_num"] += 1
            logging.debug("begin optimer,step_idx={}".format(step_idx))
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_v = common.calc_loss_V(batch, net, tgt_net.target_model, GAMMA ** REWARD_STEPS, device=device)
            loss_v.backward()
            optimizer.step()
            t3 = time.time()
            timeCount["t2t3_sum"] += t3-t2
            timeCount["t2t3_num"] += 1

            if step_idx % TARGET_NET_SYNC == 0:
                logging.info("begin sync,step_idex={},av_t1t2={},av_t2t3={},loss={}".format(step_idx,timeCount["t1t2_sum"]/timeCount["t1t2_num"],timeCount["t2t3_sum"]/timeCount["t2t3_num"],loss_v))
                tgt_net.sync()

            if step_idx % CHECKPOINT_EVERY_STEP == 0:
                # idx = step_idx // CHECKPOINT_EVERY_STEP
                torch.save(net.state_dict(), check_data_file)
                logging.info("save,step_idx={},file={}".format(step_idx,check_data_file))

            if step_idx % VALIDATION_EVERY_STEP == 0:
                res = ValidationRun(env_val,net,episodes= 1,device= device,epsilon= 0)
                data_info_dict["valid_rewards"]=res["episode_reward"]
                with open(os.path.join(saves_path,P_DATA_INFO_FILE),"w") as f:
                    json.dump(data_info_dict,f)
                for key, val in res.items():
                    writer.add_scalar(key + "_val", val, step_idx)
                logging.info("valid,step_idx={},res={}".format(step_idx,res))


                # res = validation.validation_run(env_tst, net, device=device)
                # for key, val in res.items():
                #     writer.add_scalar(key + "_test", val, step_idx)
                # res = validation.validation_run(env_val, net, device=device)
                # for key, val in res.items():
                #     writer.add_scalar(key + "_val", val, step_idx)


if __name__ == "__main__":
    log_init("../../../../tmp/ch08_train_forex_"+str(os.getpid())+".log",'INFO')
    logging.info("start")
    test()
    # test2()
