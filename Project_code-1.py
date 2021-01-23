"""
EE 526 Final Project
@author: Xiaoshi Guo
Note: This code is modified based on https://keon.io/deep-q-learning/
"""

import gym
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import datetime as dt
import math
from keras.models import Sequential
from keras.layers import Dense
tf.keras.backend.set_floatx('float64')


# build the environment 
env = gym.make("Acrobot-v1")
print(env.observation_space) 
state_size = 6
num_actions = env.action_space.n

# build the primary and target network
primary_network = Sequential([
    Dense(256, activation='relu', input_dim=state_size, dtype='float64'),
    Dense(512, activation='relu',  dtype='float64'),
    Dense(num_actions, dtype='float64')
])

# only the primary network is compiled, as this is the only network which will be trained via the Adam optimizer.
primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')

target_network = Sequential([
    Dense(256, activation='relu', input_dim=state_size, dtype='float64'),
    Dense(512, activation='relu',dtype='float64'),
    Dense(num_actions, dtype='float64')
])

# store tuples of (s_t, a, r_t, s_(t+1)) in the replay buffer
class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)

    def num_samples(self):
        return len(self._samples)

memory = Memory(1000000)

# epsilon-greedy algorithm 
def choose_action(state, primary_network, eps):
    if random.random() < eps:
        return random.randint(0, num_actions - 1)
    else:
        return np.argmax(primary_network(state.reshape(1, -1)))
    

# training process
MAX_EPSILON = 0.99
MIN_EPSILON = 0.01
LAMBDA = 0.0005
GAMMA = 0.95
BATCH_SIZE = 32
TAU = 0.08
def train(primary_network, memory, target_network=None):
    if memory.num_samples < BATCH_SIZE * 3:
        return 0
    batch = memory.sample(BATCH_SIZE)
    states = np.array([val[0] for val in batch])
    actions = np.array([val[1] for val in batch])
    rewards = np.array([val[2] for val in batch])
    next_states = np.array([(np.zeros(state_size)
                             if val[3] is None else val[3]) for val in batch])
    # predict Q(s,a) given the batch of states
    prim_qt = primary_network(states)
    # predict Q(s',a') from the evaluation network
    prim_qtp1 = primary_network(next_states)
    # copy the prim_qt tensor into the target_q tensor - we then will update one index corresponding to the max action
    target_q = prim_qt.numpy()
    updates = rewards
    valid_idxs = np.array(next_states).sum(axis=1) != 0
    batch_idxs = np.arange(BATCH_SIZE)
    if target_network is None:
        updates[valid_idxs] += GAMMA * np.amax(prim_qtp1.numpy()[valid_idxs, :], axis=1)
    else:
        prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
        q_from_target = target_network(next_states)
        updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
    target_q[batch_idxs, actions] = updates
    loss = primary_network.train_on_batch(states, target_q)
    if target_network is not None:
        # update target network parameters slowly from primary network
        for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
            t.assign(t * (1 - TAU) + e * TAU)
    return loss


# Run Episodes
num_episodes = 1000
eps = MAX_EPSILON
render = False
double_q = True
steps = 0
I =[]
CNT = []
AVG_LOSS = []
for i in range(num_episodes):
    state = env.reset()
    cnt = 0
    avg_loss = 0
    while True:
        if render:
            env.render()
        action = choose_action(state, primary_network, eps)
        next_state, reward, done, info = env.step(action)
        
        if done:
            next_state = None
        # store in memory
        memory.add_sample((state, action, reward, next_state))

        loss = train(primary_network, memory, target_network if double_q else None)
        avg_loss += loss

        state = next_state

        # exponentially decay the eps value
        steps += 1
        eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)

        if done:
            avg_loss =avg_loss/cnt
            print(f"Episode: {i}, Reward: {cnt}, avg loss: {avg_loss:.3f}, eps: {eps:.3f}")
            break

        cnt -= 1
    I.append(i)
    AVG_LOSS.append(-avg_loss)
    CNT.append(cnt)


# plot 
plt.plot(I,CNT,'grey', label='Score')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Score')


plt.plot(I,AVG_LOSS,'b-', label='Average loss')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Average loss')

primary_network.summary()




