import sys
import os
import datetime
import gym
import pybullet_envs
# from PPO import PPO
from DQN import DQN
from trainer import Trainer

from env import batteryEnv

if __name__ == '__main__':

    start_time = datetime.datetime.now()

    NUM_STEPS = 1 * 10 ** 5
    EVAL_INTERVAL = 5 * 10 ** 3

    env = batteryEnv(10, 6, 550)
    env_test = batteryEnv(10, 6, 550)

    algo = DQN(
        state_size=env.state_space.shape[0],
        action_size=env.action_space.n,
        epsilon_decay = NUM_STEPS,
        start_steps=10000,
        hidden_size=256
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        num_steps=NUM_STEPS,
        eval_interval=EVAL_INTERVAL
    )

    start_time = datetime.datetime.now()
    now = '{0:%Y%m%d}-{0:%H%M}'.format(start_time)

    trainer.train()

    trainer.plot()

    algo.save()

    end_time = datetime.datetime.now()

    # trainer.visualize()
