#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 12:51:11 2019

@author: xiaoshiguo
"""

import numpy as np
def v0(a,r,v_0,v_1):
    if a == 1:
        v_0 = 1+r*(1/3*v_0+2/3*v_1)
    else:
        v_0 = 4+r*(1/2*v_0+1/2*v_1)
    return v_0

def v1(a,r,v_0,v_1):
    if a == 1:
        v_1 = 3+r*(1/4*v_0+3/4*v_1)
    else:
        v_1 = 2+r*(2/3*v_0+1/3*v_1)
    return v_1

v_0 = 0
v_1 = 0
iteration = 100
for i in range(1,iteration):
    v_0_t = np.maximum(v0(1,0.75,v_0, v_1), v0(2,0.75,v_0,v_1))
    v_1_t = np.maximum(v1(1,0.75,v_0, v_1), v1(2,0.75,v_0, v_1))
    v_0 = v_0_t
    v_1 = v_1_t
    print("v_0 is", v_0, "and v_1 is", v_1)
    
v01 = v0(1,0.75,v_0, v_1)
v02 = v0(2,0.75,v_0,v_1)
v10 = v1(1,0.75,v_0, v_1)
v12 = v1(2,0.75,v_0, v_1)
print(v01, v02, v10, v12)