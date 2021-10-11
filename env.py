import gym
import numpy as np
import math
import random
from scipy.stats import truncnorm

class batteryEnv(gym.Env):

    def __init__(self, lift_num, battery_num, working_minutes, step_minutes=5):
        self.lift_num = lift_num
        self.battery_num = battery_num
        self.working_minutes = working_minutes
        self.step_minutes = step_minutes

        # self.state_space = gym.spaces.Box(low=0.0, high=math.inf, shape=(lift_num + battery_num + 1,))
        self.state_space = gym.spaces.Box(low=0.0, high=math.inf, shape=(lift_num + battery_num + 1+1,))
        self.action_space = gym.spaces.Discrete(min(lift_num, battery_num) + 1) 

        self.reset()

    def reset(self, test=False):
        if test:
            self.left_time = self.working_minutes
            self.lifts = np.full(self.lift_num, 100.0)
            self.batterys = np.full(self.battery_num, 100.0)
        else:
            # self.left_time = random.uniform(1, self.working_minutes//self.step_minutes) * self.step_minutes
            # self.left_time = random.uniform(1, 2*self.working_minutes//self.step_minutes) * self.step_minutes

            a, b = 1, 2*self.working_minutes/self.step_minutes
            mean = self.working_minutes
            std =  self.working_minutes / 2
            a, b = (a - mean) / std, (b - mean) / std
            self.left_time = int(round(*truncnorm.rvs(a, b, loc=mean, scale=std, size=1)))*self.step_minutes
            self.lifts = np.random.uniform(low=10.0, high=100.0, size=self.lift_num)
            self.batterys = np.random.uniform(low=10.0, high=100.0, size=self.battery_num)
        
        self.sort()

        # self.left_time = self.working_minutes
        self.done = False

        self.sum_exchange = 0
        self.old_action = 0

        return self.getState()

    def sort(self):
        self.lifts = np.sort(self.lifts)
        self.batterys = np.sort(self.batterys)[::-1]

    def step(self, action):
        self.sum_exchange += action

        change_is_odd = False
        if action > 0:
            change_is_odd = self.lifts[action-1] + 10.0 > self.batterys[action-1]

        change_over = (self.old_action + action) - self.battery_num

        change_diff = 0.0
        if action > 0:
            change_diff = self.batterys[action-1]-self.lifts[action-1]

        ex_batterys = self.lifts[:action].copy()
        self.lifts[:action] = self.batterys[:action]
        self.batterys[:action] = ex_batterys

        # self.lifts -= np.random.uniform(low=1.5, high=2.5, size=self.lift_num)
        self.lifts -= np.random.uniform(low=0.1, high=2.0, size=self.lift_num)
        lift_is_zero = np.min(self.lifts) <= 0.0
        self.lifts = np.clip(self.lifts, 0, 100)

        # self.batterys += np.random.uniform(low=4.5, high=5.5, size=self.battery_num)
        self.batterys += np.full(self.battery_num, 3.47)
        self.batterys = np.clip(self.batterys, 0, 100)

        self.sort()
       
        self.left_time += -self.step_minutes

        state = self.getState()
        reward = self.getReward(lift_is_zero, change_is_odd, change_over, change_diff)

        self.done = (lift_is_zero) or (self.left_time <= 0) or self.done

        self.old_action = action

        return state, reward, self.done, {}

    def getState(self):
        # return np.concatenate([self.lifts, self.batterys, [self.left_time]])
        return np.concatenate([self.lifts, self.batterys, [self.sum_exchange], [self.left_time]])
        # return np.concatenate([[self.old_action], self.lifts, self.batterys, [self.left_time]])

    def getReward(self, lift_is_zero, change_is_odd, change_over, change_diff):
        reward = 0
        
        if self.left_time <= 0:
            reward += -self.sum_exchange 

        elif lift_is_zero:
            reward += -10000 

        if change_is_odd:
            # reward += -1
            pass

        if change_over > 0:
            # reward += -1
            # reward += -change_over
            pass

        # reward += change_diff/100.0

        return  reward
        
    def sample_random_action(self):
        return self.action_space.sample()

if __name__ == '__main__':

    import csv

    env = batteryEnv(10, 6, 550)
    env.reset(test=False)
    done = False

    a = env.left_time
    b = env.left_time
    while True:
        env.reset(test=False)
        # a = max(a, env.left_time)
        # b = min(b, env.left_time)
        # print(a, b)
        print(env.left_time)
    # with open('env_state.csv', 'w') as f:
    #     writer = csv.writer(f)

    #     while not done:
    #         state, reward, done, _ = env.step(np.random.randint(0, 6))
    #         writer.writerow(state)