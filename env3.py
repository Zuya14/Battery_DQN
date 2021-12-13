import gym
import numpy as np
import math
import random
from scipy.stats import truncnorm

class batteryEnv3(gym.Env):

    def __init__(self, lift_num, battery_num, working_minutes, max_limit_change=None, step_minutes=5):
        self.name = "batteryEnv3"

        self.lift_num = lift_num
        self.battery_num = battery_num
        self.working_minutes = working_minutes
        self.step_minutes = step_minutes/lift_num

        self.state_space = gym.spaces.Box(low=0.0, high=math.inf, shape=(1 + lift_num + battery_num + 1,))

        # self.state_space = gym.spaces.Box(low=0.0, high=math.inf, shape=(lift_num + lift_num + battery_num + 1,))
        # self.state_space = gym.spaces.Box(low=0.0, high=math.inf, shape=(lift_num + lift_num + battery_num + 1 + 1,))
        # self.state_space = gym.spaces.Box(low=0.0, high=math.inf, shape=(lift_num + lift_num + battery_num + 1 + 1 + 1,))
        self.action_space = gym.spaces.Discrete(2) 

        if max_limit_change is None:
            self.max_limit_change = np.inf
        else:
            self.max_limit_change = max_limit_change

        self.reset()

    def reset(self, test=False):
        if test:
            self.left_time = self.working_minutes
            self.lifts = np.full(self.lift_num, 100.0)
            self.batterys = np.full(self.battery_num, 100.0)
        else:
            self.left_time = self.working_minutes
            # self.left_time = random.uniform(1, self.working_minutes//self.step_minutes) * self.step_minutes
            # self.lifts = np.full(self.lift_num, 100.0)
            # self.batterys = np.full(self.battery_num, 100.0)
            self.lifts = np.random.uniform(low=10.0, high=100.0, size=self.lift_num)
            self.batterys = np.random.uniform(low=10.0, high=100.0, size=self.battery_num)

        # self.sort()

        # self.left_time = self.working_minutes
        self.done = False

        self.sum_exchange = 0
        self.old_action = 0

        self.old_change_time = 0

        self.index = 0
        self.label = self.getLabel(self.index)

        return self.getState()

    def getLabel(self, index):
        label = np.zeros(self.lift_num)
        label[index] = 1
        return label

    def sort(self):
        # self.lifts = np.sort(self.lifts)
        self.batterys = np.sort(self.batterys)[::-1]

    def step(self, action):

        _action = action

        if self.sum_exchange >= self.max_limit_change:
            action = 0
    
        if action > 0:
            self.sum_exchange += 1

            ex_batterys = self.lifts.copy()
            self.lifts[self.index] = self.batterys[0]
            self.batterys[0] = ex_batterys[self.index]

        self.lifts -= np.random.uniform(low=0.1, high=2.0, size=self.lift_num) * (self.step_minutes/5)
        lift_is_zero = np.min(self.lifts) <= 0.0
        self.lifts = np.clip(self.lifts, 0, 100)

        self.batterys += np.full(self.battery_num, 3.47) * (self.step_minutes/5)
        self.batterys = np.clip(self.batterys, 0, 100)

        self.sort()
       
        self.left_time += -self.step_minutes
        self.index = (self.index + 1) % self.lift_num

        self.label = self.getLabel(self.index)

        state = self.getState()
        reward = self.getReward(lift_is_zero, _action)

        # self.done = (lift_is_zero) or (self.left_time <= 0) or self.done
        self.done = (self.left_time <= 0) or self.done

        self.old_action = action

        return state, reward, self.done, {}

    def getState(self):
        # return np.concatenate([self.label, self.lifts/100.0, self.batterys/100.0, [self.sum_exchange], [self.left_time/self.working_minutes]])
        return np.concatenate([[self.lifts[self.index]/100.0], self.lifts/100.0, self.batterys/100.0, [self.left_time/self.working_minutes]])
        # return np.concatenate([self.label, self.lifts/100.0, self.batterys/100.0, [self.left_time/self.working_minutes]])
        # return np.concatenate([self.label, self.lifts/100.0, self.batterys/100.0, [np.min(self.lifts)/100.0], [self.sum_exchange], [self.left_time/self.working_minutes]])
        # return np.concatenate([self.label, self.lifts/100.0, self.batterys/100.0, [np.min(self.batterys)/100.0], [self.sum_exchange], [math.log(1+self.left_time)]])

    def getReward(self, lift_is_zero, action):
        reward = 0
        
        if self.left_time <= 0:
            # reward += 1.0 - self.sum_exchange/self.max_limit_change 
            # reward += 1.0 - self.sum_exchange/1100 
            # reward += -self.sum_exchange 
            # reward += 1.0 / (1+self.sum_exchange) 
            reward += 1.005 ** (-self.sum_exchange) 
            # reward += -self.sum_exchange * 10
            # reward += (660-self.sum_exchange)/660

        elif lift_is_zero:
            # pass
            # reward += -1.0
            reward += -np.count_nonzero(self.lifts <= 0.0)
            # reward += -10000 
            # reward += -1000000 
            # reward += -10000 * (1 - (np.sum(self.lifts) / (100.0*self.lift_num)))
        
        elif action == 0:
            # reward += 1.0/self.lift_num
            # reward += 1.0
            pass

        elif action == self.old_action:
            # reward += -1/self.lift_num
            # reward += -10.0
            pass

        return  reward
        
    def sample_random_action(self):
        return self.action_space.sample()

if __name__ == '__main__':

    import csv

    env = batteryEnv3(10, 6, 550)
    env.reset(test=False)
    done = False

    env.reset(test=False)
    with open('env3_state.csv', 'w', newline="") as f:
        writer = csv.writer(f)

        while not done:
            state, reward, done, _ = env.step(env.sample_random_action())
            writer.writerow(state)