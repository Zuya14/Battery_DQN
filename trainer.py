import os
import glob
from time import time
from datetime import timedelta
from base64 import b64encode
import numpy as np
import gym
import matplotlib.pyplot as plt
import csv
import sys

aaa = 62

class Trainer:

    def __init__(self, env, env_test, algo, seed=0, num_steps=10**6, eval_interval=10**4, num_eval_episodes=1):

        self.env = env
        self.env_test = env_test
        self.algo = algo

        # 環境の乱数シードを設定する．
        self.env.seed(seed)
        self.env_test.seed(2**31-seed)

        # 平均収益を保存するための辞書．
        self.returns = {'step': [], 'return': [],  'sum_exchange': []}

        # データ収集を行うステップ数．
        self.num_steps = num_steps
        # 評価の間のステップ数(インターバル)．
        self.eval_interval = eval_interval
        # 評価を行うエピソード数．
        self.num_eval_episodes = num_eval_episodes
    
    def train(self, path):
        """ num_stepsステップの間，データ収集・学習・評価を繰り返す． """

        # 学習開始の時間
        self.start_time = time()
        # エピソードのステップ数．
        t = 0

        # 環境を初期化する．
        state = self.env.reset()

        for steps in range(1, self.num_steps + 1):
            # 環境(self.env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            # アルゴリズムに渡し，状態・エピソードのステップ数を更新する．

            state, t = self.algo.step(self.env, state, t, steps)

            # アルゴリズムが準備できていれば，1回学習を行う．
            if self.algo.is_update(steps):
                self.algo.update()

            # 一定のインターバルで評価する．
            if steps % self.eval_interval == 0:
                self.evaluate(steps, path)

    def evaluate(self, steps, path):
        """ 複数エピソード環境を動かし，平均収益を記録する． """

        path = os.path.join(path, str(steps))
        if not os.path.exists(path):
            os.makedirs(path)

        csv_path = os.path.join(path, "q.csv")
        csv_path2 = os.path.join(path, "state.csv")
        csv_path3 = os.path.join(path, "action.csv")

        with open(csv_path, 'w', newline="") as f:
            writer = csv.writer(f)
            with open(csv_path2, 'w', newline="") as f2:
                writer2 = csv.writer(f2)
                with open(csv_path3, 'w', newline="") as f3:
                    writer3 = csv.writer(f3)
                    
                    returns = []
                    for _ in range(self.num_eval_episodes):
                        state = self.env_test.reset(test=True)
                        done = False
                        episode_return = 0.0

                        while not done:
                            action, q = self.algo.exploit2(state)
                            state, reward, done, _ = self.env_test.step(action)
                            episode_return += reward

                            # writer.writerow(q)
                            writer.writerow([*q,  np.average(q), *(q-np.average(q))])
                            writer2.writerow(state)
                            writer3.writerow([action])
                            if done:
                                break

                        returns.append(episode_return)
        
        mean_return = np.mean(returns)
        self.returns['step'].append(steps)
        self.returns['return'].append(mean_return)
        self.returns['sum_exchange'].append(self.env_test.sum_exchange)

        print(f'Num steps: {steps:<6}   '
            f'Return: {mean_return:<5.4f}   '
            f'sum_exchange: {self.env_test.sum_exchange:<5.1f}   '
            #   f'Final state: {state}   '
            f'Final_minute: {self.env_test.left_time}   '
            f'Time: {self.time}')
        sys.stdout.flush()

    def plot(self, path="./", s=""):
        """ 平均収益のグラフを描画する． """
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.returns['step'], self.returns['return'])
        plt.xlabel('Steps', fontsize=24)
        plt.ylabel('Return', fontsize=24)
        plt.tick_params(labelsize=18)
        # plt.title(f'{self.env.unwrapped.spec.id}', fontsize=24)
        plt.title('return', fontsize=24)
        plt.tight_layout()
        # fig.savefig("log/"+self.algo.name+s+".png")
        fig.savefig(os.path.join(path, self.algo.name + s + "_return.png"))

        fig = plt.figure(figsize=(8, 6))
        plt.plot([i for i in range(self.eval_interval, self.eval_interval+len(self.algo.log_loss), 100)], self.algo.log_loss[::100])
        plt.xlabel('Steps', fontsize=24)
        plt.ylabel('Loss', fontsize=24)
        plt.tick_params(labelsize=18)
        # plt.title(f'{self.env.unwrapped.spec.id}', fontsize=24)
        plt.title('loss', fontsize=24)
        plt.tight_layout()
        # fig.savefig("log/"+self.algo.name+s+".png")
        # fig.savefig(path + f"/{aaa}/" + self.algo.name + s + "_loss.png")
        fig.savefig(os.path.join(path, self.algo.name + s + "_loss.png"))

        # path = f'./{aaa}/' 
        if not os.path.exists(path):
            os.makedirs(path)

        csv_path = os.path.join(path, "return.csv")
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            for r in np.array(self.returns['return']):
                writer.writerow([r])

        csv_path = os.path.join(path, "sum_exchange.csv")
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            for r in np.array(self.returns['sum_exchange']):
                writer.writerow([r])

    @property
    def time(self):
        """ 学習開始からの経過時間． """
        return str(timedelta(seconds=int(time() - self.start_time)))
