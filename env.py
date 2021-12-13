import gym
import numpy as np
import math
import random
from scipy.stats import truncnorm

class batteryEnv(gym.Env):

    def __init__(self, lift_num, battery_num, working_minutes, max_limit_change=None, step_minutes=5):
        self.name = "batteryEnv"

        self.lift_num = lift_num
        self.battery_num = battery_num
        self.working_minutes = working_minutes
        self.step_minutes = step_minutes

        # self.state_space = gym.spaces.Box(low=0.0, high=math.inf, shape=(lift_num + battery_num + 1,))
        # self.state_space = gym.spaces.Box(low=0.0, high=math.inf, shape=(lift_num + battery_num + 1+1,))
        self.state_space = gym.spaces.Box(low=0.0, high=math.inf, shape=((lift_num + battery_num)*2 + 1+1,))
        # self.state_space = gym.spaces.Box(low=0.0, high=math.inf, shape=(lift_num + battery_num + 1+1+1,))
        # self.state_space = gym.spaces.Box(low=0.0, high=math.inf, shape=(lift_num + battery_num + 1+1+1+1,))
        self.action_space = gym.spaces.Discrete(min(lift_num, battery_num) + 1) 

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
            # self.left_time = random.uniform(1, 2*self.working_minutes//self.step_minutes) * self.step_minutes

            # a, b = 1, 2*self.working_minutes/self.step_minutes
            # mean = self.working_minutes
            # std =  self.working_minutes / 2
            # a, b = (a - mean) / std, (b - mean) / std
            # self.left_time = int(round(*truncnorm.rvs(a, b, loc=mean, scale=std, size=1)))*self.step_minutes
            self.lifts = np.random.uniform(low=10.0, high=100.0, size=self.lift_num)
            self.batterys = np.random.uniform(low=10.0, high=100.0, size=self.battery_num)
            # self.lifts = np.full(self.lift_num, 100.0)
            # self.batterys = np.full(self.battery_num, 100.0)
        
        self.old_lifts = self.lifts.copy()
        self.old_batterys = self.batterys.copy()
        self.lifts_diff = np.zeros(self.lift_num)
        self.batterys_diff = np.zeros(self.battery_num)

        self.sort()

        # self.left_time = self.working_minutes
        self.done = False

        self.sum_exchange = 0
        self.old_action = 0

        self.old_change_time = 0

        return self.getState()

    def sort(self):
        self.lifts = np.sort(self.lifts)
        self.batterys = np.sort(self.batterys)[::-1]

    def step(self, action):

        _action = action

        if action + self.sum_exchange >= self.max_limit_change:
            action = self.max_limit_change - self.sum_exchange
    
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
        self.lifts -= np.random.uniform(low=0.1, high=2.0, size=self.lift_num) * (self.step_minutes/5)
        lift_is_zero = np.min(self.lifts) <= 0.0
        self.lifts = np.clip(self.lifts, 0, 100)

        # self.batterys += np.random.uniform(low=4.5, high=5.5, size=self.battery_num)
        self.batterys += np.full(self.battery_num, 3.47) * (self.step_minutes/5)
        self.batterys = np.clip(self.batterys, 0, 100)

        self.sort()
       
        self.lifts_diff = self.lifts - self.old_lifts
        self.batterys_diff = self.batterys - self.old_batterys

        self.old_lifts = self.lifts.copy()
        self.old_batterys = self.batterys.copy()


        self.left_time += -self.step_minutes

        state = self.getState()
        reward = self.getReward(lift_is_zero, change_is_odd, change_over, change_diff, _action) 

        # self.done = (lift_is_zero) or (self.left_time <= 0) or self.done
        self.done = (self.left_time <= 0) or self.done

        self.old_action = action

        if action > 0:
            self.old_change_time = 0
        else:
            self.old_change_time += self.step_minutes

        return state, reward, self.done, {}

    def getState(self):
        # return np.concatenate([self.lifts, self.batterys, [self.left_time]])
        # return np.concatenate([self.lifts, self.batterys, [self.sum_exchange], [self.left_time]])
        # return np.concatenate([self.lifts, self.batterys, [self.old_change_time], [self.sum_exchange], [self.left_time]])
        # return np.concatenate([self.lifts, self.batterys, [self.old_action], [self.old_change_time], [self.sum_exchange], [self.left_time]])
        # return np.concatenate([[self.old_action], self.lifts, self.batterys, [self.left_time]])
        # return np.concatenate([self.lifts/100.0, self.batterys/100.0, [self.sum_exchange/self.max_limit_change ], [self.left_time/self.working_minutes]])
        # return np.concatenate([self.lifts/100.0, self.batterys/100.0, [self.sum_exchange/self.max_limit_change ], [self.left_time<self.working_minutes/10]])
        # return np.concatenate([self.lifts/100.0, self.batterys/100.0, [self.sum_exchange/self.max_limit_change ]])

        return np.concatenate([self.lifts, self.batterys, self.lifts_diff, self.batterys_diff, [self.sum_exchange], [self.left_time]])

    def getReward(self, lift_is_zero, change_is_odd, change_over, change_diff, action):
        reward = 0
        # reward += -100*action
        if self.left_time <= 0:
            # pass
            reward += 1.0 / (1+self.sum_exchange) 
            # reward += 1.0 - self.sum_exchange/self.max_limit_change 
            # reward += -self.sum_exchange 
            # reward += -self.sum_exchange * 10
            # reward += (660-self.sum_exchange)/660

        elif lift_is_zero:
            # pass
            reward += -np.count_nonzero(self.lifts <= 0.0)
            # reward += -1
            # reward += -10000 
            # reward += -1000000 
            # reward += -10000 * (1 - (np.sum(self.lifts) / (100.0*self.lift_num)))

        elif action == 0:
            # reward += 10
            # reward += 0.1
            # reward += 0.01
            # reward += 0.0001
            pass
        elif action != 0:
            # reward += -0.1*action
            # reward += -0.01
            # reward += -1.0*action/min(self.lift_num, self.battery_num)
            pass

            # if self.sum_exchange >= self.max_limit_change:
            #     reward += -0.01

        if change_is_odd:
            # reward += -1
            pass

        if change_over > 0:
            # reward += -1
            # reward += -change_over
            pass

        # reward += change_diff/100.0

        # reward = reward / 10000

        # reward += - np.mean(np.where(self.batterys_diff>0, self.batterys_diff, 0))/100

        return  reward
        
    def sample_random_action(self):
        return self.action_space.sample()

if __name__ == '__main__':

    import csv

    env = batteryEnv(10, 6, 550)
    env.reset(test=False)
    done = False

    env.reset(test=False)
    with open('env_state.csv', 'w', newline="") as f:
        writer = csv.writer(f)

        while not done:
            state, reward, done, _ = env.step(env.sample_random_action())
            writer.writerow(state)