import sys

sys.path.append("..")


work_dir ="/home/sailingzhang/winshare/develop/source/mygit"
sys.path.append(work_dir)
sys.path.append(work_dir+"\\test\\python")
sys.path.append("../../ptan-master")

import logging
from log_init import log_init

import gym
import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.StockTradingEnv import StockTradingEnv

import pandas as pd


from gym_trading.envs.forex_env import forex_candle_env,ValidationRun


FOREX_DATA_PATH=work_dir+"/gym_trading/data/FOREX_EURUSD_1H_ASK_CLOSE.csv"




def test():
    df = pd.read_csv('./data/AAPL.csv')
    df = df.sort_values('Date')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=20000)

    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

def test2():
    env = forex_candle_env(FOREX_DATA_PATH, window_size=600,initCapitalPoint=2000,feePoint=20)
    env_val = forex_candle_env(FOREX_DATA_PATH, window_size=600,initCapitalPoint=2000,feePoint=20)
    env = DummyVecEnv([lambda: env])
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=200000)
    logging.info("learn over")
    # ValidationRun(env_val,net,episodes= 1,device= device,epsilon= 0)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        logging.info("baseline info={}".format(info))
        if done:
            logging.info("game over")
            return

if __name__ == "__main__":
    log_init("/tmp/stock.log","INFO")
    logging.info("start openai")
    test2()
