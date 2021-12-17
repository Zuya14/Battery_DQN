import numpy as np
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from env import batteryEnv

# ENV_NAME = 'CartPole-v0'


# Get the environment and extract the number of actions.
# env = gym.make(ENV_NAME)
env_time = 550
max_limit_change = None
step_minutes = 5
lift_num = 3
battery_num = 1
env = batteryEnv(lift_num, battery_num, env_time, max_limit_change, step_minutes)
# np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n

ENV_NAME = env.name

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

STEPS = 10**5
# STEPS = 10**3

# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=10**6, window_length=1)
# policy = BoltzmannQPolicy()
_policy = EpsGreedyQPolicy()

for i in range(0, 10):
    # policy = LinearAnnealedPolicy(_policy, attr='eps', value_max=1.0, value_min=0.01, value_test=0.0, nb_steps=STEPS)
    policy = LinearAnnealedPolicy(_policy, attr='eps', value_max=1.0*(10-i), value_min=max(1.0*(9-i), 0.01), value_test=0.0, nb_steps=STEPS//10)

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                    # nb_steps_warmup=10,
                    nb_steps_warmup=10 if i < 1 else 0,
                    target_model_update=1e-2, 
                    policy=policy,
                    batch_size=64)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=STEPS//10, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights(f'dqn_{ENV_NAME}_weights.h5f', overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=1, visualize=False)
