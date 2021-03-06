import torch
import torch.nn as nn
import torch.nn.functional as F
# from ReplayBuffer import ReplayBuffer
from cpprb import ReplayBuffer, PrioritizedReplayBuffer
import random
import numpy as np

class Net(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            # nn.ELU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            # nn.ELU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.net(x)

class DNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            # nn.ELU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            # nn.ELU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        # self.net = nn.Sequential(
        #     nn.Linear(state_size, hidden_size),
        #     nn.ELU(inplace=True),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ELU(inplace=True),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ELU(inplace=True)
        # )

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
                 batch_size=64, gamma=0.99, lr=1e-3,
                #  batch_size=256, gamma=0.99, lr=1e-3,
                 replay_size=10**6, start_steps=10**4, tau=5e-3, alpha=0.2, reward_scale=1.0, epsilon_decay = 50000,
                 action_repeat=1):

        super().__init__()

        self.name = 'DQN'

        # ???????????????????????????
        # self.buffer = ReplayBuffer(
        #     buffer_size=replay_size,
        #     # buffer_size=epsilon_decay//2,
        #     state_size=state_size,
        #     action_size=action_size,
        #     device=device
        # )


        N_iteration = epsilon_decay
        # nstep = 3

        # if nstep:
        #     Nstep = {"size": nstep, "gamma": gamma, "rew": "rew", "next": "next_obs"}

        env_dict = {"obs":{"shape": state_size},
                    "act":{"shape": 1,"dtype": np.ubyte},
                    "rew": {},
                    "next_obs": {"shape": state_size},
                    "done": {}}

        self.buffer = PrioritizedReplayBuffer(
            replay_size,
            env_dict,
            # Nstep=Nstep
            )

        # Beta linear annealing
        self.beta = 0.4
        self.beta_step = (1 - self.beta)/N_iteration


        # ????????????????????????????????????
        self.QNet = QNetwork(state_size, action_size, hidden_size).to(device)
        self.QNet_target = QNetwork(state_size, action_size, hidden_size).to(device).eval()

        # ?????????????????????????????????????????????????????????????????????????????????????????????
        self.QNet_target.load_state_dict(self.QNet.state_dict())
        for param in self.QNet_target.parameters():
            param.requires_grad = False

        # ????????????????????????
        self.optim_QNet = torch.optim.Adam(self.QNet.parameters(), lr=lr)

        # ???????????????????????????
        self.action_size = action_size
        self.learning_steps = 0
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.start_steps = start_steps
        self.tau = tau
        self.alpha = alpha
        self.reward_scale = reward_scale

        self.action_repeat = action_repeat

        epsilon_begin = 1.0
        epsilon_end = 0.01
        # epsilon_begin???????????????epsilon_end??????epsilon_decay???????????????????????????
        self.epsilon_func = lambda step: max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))

        self.log_loss = []

    def is_update(self, steps):
        # ???????????????????????????(start_steps)?????????????????????
        return steps >= max(self.start_steps, self.batch_size)

    def exploit(self, state):
        """ ????????????????????????????????? """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = torch.argmax(torch.min(*self.QNet(state)).data).item()
        return action

    def exploit2(self, state):
        """ ????????????????????????????????? """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            q = torch.min(*self.QNet(state)).data
            action = torch.argmax(q).item()
        return action, q.cpu().numpy().flatten()

    def step(self, env, state, t, steps):

        # ???????????????????????????(start_steps)??????????????????????????????????????????????????????????????????????????????
        if steps <= self.start_steps:
            action = env.action_space.sample()
        else:
            if random.random() < self.epsilon_func(steps):
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = self.exploit(state)

        # if steps <= self.start_steps:
        #     if random.random() < 0.5:
        #         action = 0
        #     else:
        #         action = env.action_space.sample()
        # else:
        #     if random.random() < self.epsilon_func(steps):
        #         if random.random() < 0.5:
        #             action = 0
        #         else:
        #             action = env.action_space.sample()
        #     else:
        #         with torch.no_grad():
        #             action = self.exploit(state)

        for _ in range(self.action_repeat):
            t += 1

            next_state, reward, done, _ = env.step(action)

            # ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
            # ?????????????????????????????????????????????????????????????????????????????????????????????False????????????
            # NOTE: ??????????????????????????????????????????????????????????????????????????? done_masked=True ????????????
            # ????????????????????????????????????"????????????"????????????????????????????????????????????????????????????????????????
            # done_masked=False ????????????????????????
            # ???????????????????????????????????????????????????????????????????????????????????????????????????????????????
            # if t == env._max_episode_steps:
            # if 0 == env.left_time:
            #     done_masked = False
            # else:
            #     done_masked = done
            done_masked = done

            # ??????????????????????????????????????????????????????
            # self.buffer.append(state, action, reward, done_masked, next_state)
            self.buffer.add(obs=state,
                act=action,
                rew=reward,
                next_obs=next_state,
                done=done_masked)

            # ???????????????????????????????????????????????????????????????????????????
            if done:
                t = 0
                next_state = env.reset()
                self.buffer.on_episode_end()

                # if steps <= self.start_steps:
                #     action = env.action_space.sample()
                # else:
                #     if random.random() < self.epsilon_func(steps):
                #         action = env.action_space.sample()
                #     else:
                #         with torch.no_grad():
                #             action = self.exploit(state)

        return next_state, t

    def update(self, writer):
        self.learning_steps += 1
        # states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        sample = self.buffer.sample(self.batch_size, self.beta)
        self.beta += self.beta_step

        # self.update_QNet(states, actions, rewards, dones, next_states)
        self.update_QNet(sample, writer)
        self.update_target()

    # def update_QNet(self, states, actions, rewards, dones, next_states):
    def update_QNet(self, sample, writer):
        states, actions, rewards, dones, next_states = sample["obs"], sample["act"].ravel(), sample["rew"].ravel(), sample["done"].ravel(), sample["next_obs"]
        states = torch.from_numpy(states).view(self.batch_size, -1).to(self.device)
        actions = torch.from_numpy(actions).view(self.batch_size, -1).to(self.device)
        rewards = torch.from_numpy(rewards).view(self.batch_size, -1).to(self.device)
        dones = torch.from_numpy(dones).view(self.batch_size, -1).to(self.device)
        next_states = torch.from_numpy(next_states).view(self.batch_size, -1).to(self.device)

        curr_qs1, curr_qs2 = self.QNet(states)

        with torch.no_grad():
            next_qs1, next_qs2 = self.QNet_target(next_states)
            next_qs = torch.min(next_qs1, next_qs2)
        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        absTD = (torch.abs(curr_qs1 - target_qs) + torch.abs(curr_qs2 - target_qs)) * 0.5
        weights = torch.from_numpy(sample["weights"].ravel()).view(self.batch_size, -1).to(self.device)

        # loss_QNet1 = ((curr_qs1 - target_qs)*weights).pow_(2).mean()
        # loss_QNet2 = ((curr_qs2 - target_qs)*weights).pow_(2).mean()

        huber = nn.SmoothL1Loss(reduction='none')
        # loss_QNet1 = (huber(curr_qs1, target_qs)).mean()
        # loss_QNet2 = (huber(curr_qs2, target_qs)).mean()
        loss_QNet1 = (huber(curr_qs1, target_qs)*weights).mean()
        loss_QNet2 = (huber(curr_qs2, target_qs)*weights).mean()

        self.optim_QNet.zero_grad()
        loss = loss_QNet1 + loss_QNet2

        loss.backward(retain_graph=False)
        self.optim_QNet.step()

        self.log_loss.append(loss.item())

        # absTD = loss/2
        absTD = absTD.detach().cpu().numpy()
        self.buffer.update_priorities(sample["indexes"], absTD)

        writer.add_scalar("loss/q", loss.item(), self.learning_steps)

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

    print(dqn.exploit(torch.FloatTensor(5).uniform_(10, 100)))
