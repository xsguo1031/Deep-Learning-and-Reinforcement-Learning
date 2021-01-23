#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:20:50 2019

@author: xiaoshiguo
"""
#----------------Part a------------
import numpy as np
import matplotlib.pyplot as plt


state = np.random.randint(2, size=10000)
state[0] = 0 
action = state +1
reward = np.zeros(state.shape) #initialize reward
reward[1:] = state[0:-1] + 1 #if last state is 0 then reward is 1, if last state is 1 then reward is 2
for i in range(0, 10000):
    print("reward is ", reward[i], "state is ", state[i],  "and action is ", action[i])


#------------ Part b---------------- 
# initial value fucntions
v0 = 0
v1 = 0
# initial number of times we visit state 0 and state 1
n0 = 0
n1 = 0
# calculate the G for each time t 
def GT(state, reward):
    Gt = 0
    for i in range(1, len(state)):
        Gt = Gt + reward[i] * np.power(0.75, i-1) 
    return Gt

# estimate v0 and v1 using MC 
for i in range(0, 10000):
    if state[i] == 0:
        n0 = n0+1
        gt = GT(state[i:], reward[i:])
        v0 = v0 + 1/n0 *(gt-v0)
    else: 
        n1 = n1+1
        gt = GT(state[i:], reward[i:])
        v1 = v1 + 1/n1 *(gt-v1)
print("v0 is ", v0, "v1 is", v1)



#---------- Part c------------
# estimate v0 and v1 using Bootstrap
V1 = []
V0 = []
step = []
for n in range(1,50):
    # initial value fucntions
    v0 = 0
    v1 = 0
    # initial number of times we visit state 0 and state 1
    n0 = 0
    n1 = 0
    for i in range(0, 10000-n):
        if state[i] == 0:
            n0 = n0+1
            if state[i+n] == 0:
                vin = v0
            else:
                vin = v1
            gn = GT(state[i:(i+n)], reward[i:(i+n)])+0.75**n*vin    
            v0 = v0 + 1/n0 *(gn-v0)
        else: 
            n1 = n1+1
            if state[i+n] == 0:
                vin = v0
            else:
                vin = v1
            gn = GT(state[i:(i+n)], reward[i:(i+n)])+0.75**n*vin
            v1 = v1 + 1/n1 *(gn-v1)
    print("when n is ", n, "v0 is ", v0, "v1 is", v1)
    step.append(n)
    V0.append(v0)
    V1.append(v1)
plt.plot(step,V0,'r-', label='V0')
plt.plot(step,V1,'b-', label='V1')
plt.legend()
plt.xlabel('n')
plt.ylabel('Value function')
    