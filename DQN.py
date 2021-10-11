import torch
import torch.nn as nn
import torch.nn.functional as F
from ReplayBuffer import ReplayBuffer
import random

class Net(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.net(x)

class DNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=True)
        )

        self.fc_adv = nn.Linear(hidden_size, action_size)
        self.fc_v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.net(x)
        adv = self.fc_adv(h)
        val = self.fc_v(h).expand(-1, adv.size(1))
        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return output

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()

        # self.net1 = Net(state_size, action_size, hidden_size)
        # self.net2 = Net(state_size, action_size, hidden_size)

        self.net1 = DNet(state_size, action_size, hidden_size)
        self.net2 = DNet(state_size, action_size, hidden_size)

    def forward(self, x):
        return self.net1(x), self.net2(x)

class DQN:

    def __init__(self, state_size, action_size, hidden_size=64, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 batch_size=256, gamma=0.99, lr=1e-3,
                 replay_size=10**6, start_steps=10**4, tau=5e-3, alpha=0.2, reward_scale=1.0, epsilon_decay = 50000):

        super().__init__()

        self.name = 'DQN'

        # リプレイバッファ．
        self.buffer = ReplayBuffer(
            buffer_size=replay_size,
            # buffer_size=epsilon_decay//2,
            state_size=state_size,
            action_size=action_size,
            device=device
        )

        # ネットワークを構築する．
        self.QNet = QNetwork(state_size, action_size, hidden_size).to(device)
        self.QNet_target = QNetwork(state_size, action_size, hidden_size).to(device).eval()

        # ターゲットネットワークの重みを初期化し，勾配計算を無効にする．
        self.QNet_target.load_state_dict(self.QNet.state_dict())
        for param in self.QNet_target.parameters():
            param.requires_grad = False

        # オプティマイザ．
        self.optim_QNet = torch.optim.Adam(self.QNet.parameters(), lr=lr)

        # その他パラメータ．
        self.action_size = action_size
        self.learning_steps = 0
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.start_steps = start_steps
        self.tau = tau
        self.alpha = alpha
        self.reward_scale = reward_scale

        epsilon_begin = 1.0
        epsilon_end = 0.01
        # epsilon_beginから始めてepsilon_endまでepsilon_decayかけて線形に減らす
        self.epsilon_func = lambda step: max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))

        self.log_loss = []

    def is_update(self, steps):
        # 学習初期の一定期間(start_steps)は学習しない．
        return steps >= max(self.start_steps, self.batch_size)

    def exploit(self, state):
        """ 決定論的な行動を返す． """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = torch.argmax(torch.min(*self.QNet(state)).data).item()
        return action

    def exploit2(self, state):
        """ 決定論的な行動を返す． """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            q = torch.min(*self.QNet(state)).data
            action = torch.argmax(q).item()
        return action, q.cpu().numpy().flatten()

    def step(self, env, state, t, steps):
        t += 1

        # 学習初期の一定期間(start_steps)は，ランダムに行動して多様なデータの収集を促進する．
        if steps <= self.start_steps:
            action = env.action_space.sample()
        else:
            if random.random() < self.epsilon_func(steps):
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = self.exploit(state)

        next_state, reward, done, _ = env.step(action)

        # ゲームオーバーではなく，最大ステップ数に到達したことでエピソードが終了した場合は，
        # 本来であればその先も試行が継続するはず．よって，終了シグナルをFalseにする．
        # NOTE: ゲームオーバーによってエピソード終了した場合には， done_masked=True が適切．
        # しかし，以下の実装では，"たまたま"最大ステップ数でゲームオーバーとなった場合には，
        # done_masked=False になってしまう．
        # その場合は稀で，多くの実装ではその誤差を無視しているので，今回も無視する．
        # if t == env._max_episode_steps:
        # if 0 == env.left_time:
        #     done_masked = False
        # else:
        #     done_masked = done
        done_masked = done

        # リプレイバッファにデータを追加する．
        self.buffer.append(state, action, reward, done_masked, next_state)

        # エピソードが終了した場合には，環境をリセットする．
        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        self.update_QNet(states, actions, rewards, dones, next_states)
        self.update_target()

    def update_QNet(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.QNet(states)

        with torch.no_grad():
            next_qs1, next_qs2 = self.QNet_target(next_states)
            next_qs = torch.min(next_qs1, next_qs2)
        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_QNet1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_QNet2 = (curr_qs2 - target_qs).pow_(2).mean()

        # huber = nn.SmoothL1Loss()
        # loss_QNet1 = huber(curr_qs1, target_qs)
        # loss_QNet2 = huber(curr_qs2, target_qs)

        self.optim_QNet.zero_grad()
        loss = loss_QNet1 + loss_QNet2
        loss.backward(retain_graph=False)
        self.optim_QNet.step()

        self.log_loss.append(loss.item())

    def update_target(self):
        for t, s in zip(self.QNet_target.parameters(), self.QNet.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)

    def save(self, path="./"):
        torch.save(self.QNet.to('cpu').state_dict(), path+"QNet.pth")
        self.QNet.to(self.device)

    def load(self, path="./"):
        self.QNet.load_state_dict(torch.load(path+"QNet.pth"))

if __name__ == '__main__':
    dqn = DQN(5, 5)

    import numpy as np
    print(dqn.exploit(torch.FloatTensor(5).uniform_(10, 100)))
