import sys
import os
import datetime
import gym
import pybullet_envs
# from PPO import PPO
from DQN import DQN
from trainer import Trainer

from env import batteryEnv
from env2 import batteryEnv2

if __name__ == '__main__':

    start_time = datetime.datetime.now()

    # NUM_STEPS = 1 * 10 ** 6
    NUM_STEPS = 1 * 10 ** 5
    EVAL_INTERVAL = 5 * 10 ** 3

    env_time = 550
    # env_time = 225

    # max_limit_change = 50
    # max_limit_change = 100
    max_limit_change = None
    # step_minutes = 50
    step_minutes = 5

    env = batteryEnv(10, 6, env_time, max_limit_change, step_minutes)
    env_test = batteryEnv(10, 6, env_time, max_limit_change, step_minutes)
    # env = batteryEnv2(10, 6, env_time, max_limit_change, step_minutes)
    # env_test = batteryEnv2(10, 6, env_time, max_limit_change, step_minutes)


    algo = DQN(
        state_size=env.state_space.shape[0],
        action_size=env.action_space.n,
        epsilon_decay = NUM_STEPS,
        start_steps=10000,
        # hidden_size=256,
        hidden_size=64,
        # action_repeat=10
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
    path = "log/{:s}/{:s}/{:s}/".format(algo.name, env.name, now)

    if not os.path.exists(path):
        os.makedirs(path)
    print("dirs:" + path)

    trainer.train(path)

    trainer.plot(path)

    algo.save(path)

    end_time = datetime.datetime.now()

    # trainer.visualize()
