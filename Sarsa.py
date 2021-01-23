#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 21:27:36 2019

@author: xiaoshiguo
"""

import numpy as np
## Initialize action-value function
## q(s, a)
q = np.array([
        [0., 0],
        [0, 0]
        ])

## Reward(s, a)
reward = np.array([
        [1., 4.],
        [3., 2.]
        ])
## epsilon-greedy
epsilon = 0.1

## step size
alpha = 0.4

## discount factor
gamma = 0.75

#np.random.seed(1)

## Initialize state, 0 or 1
state = np.random.randint(2)
## Choose action from state with epsilon-greedy
if np.random.rand() < epsilon:
    # Explore, action = 1 or 2
    action = np.random.randint(2)
else:
    # Choose action with larger q value
    # index is 0 and 1 while action is 1 and 2
    action = np.argmax(q[state, :])

## Repeat for each step of episode 
q01 = []
q02 =[]
q11 = []
q12 = []
I =[]

for i in range(3000): 
    R = reward[state, action]
    state_next = np.random.randint(2) ## How to determine next state?
    ## Choose action_next with epsilon-greedy
    if np.random.rand() < epsilon:
        action_next = np.random.randint(2)
    else:
        action_next = np.argmax(q[state_next, :])
    ## update q value    
    q[state, action] += alpha * ( R + 
                 gamma * q[state_next, action_next] - q[state, action] )
    if i%1000 == 0:
        print("q = \n (%0.3f, %0.3f) \n (%0.3f, %0.3f)" 
              %(q[0,0],q[0, 1],q[1,0], q[1,1]))
   
    state = state_next
    action = action_next
    I.append(i)
    q01.append(q[0,0])
    q02.append(q[0,1])
    q11.append(q[1,0])
    q12.append(q[1,1])
plt.plot(I,q01,'r-', label='q01')
plt.plot(I,q02,'b-', label='q02')
plt.plot(I,q11,'y-', label='q11')
plt.plot(I,q12,'p-', label='q12')

plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Action Value Q')
