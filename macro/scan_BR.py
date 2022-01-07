#! /usr/bin/env python

import os, multiprocessing
import copy
import math
import uproot
import root_numpy
import numpy as np

def eq(H,Z):
    return H**2 + 2*H*Z + Z**2

def retZ(H):
    #x**2 + 2*H*x + (H**2 - 1.) = 0
    return -H + 1

for h in np.linspace(0,1,5):
    z = retZ(h)
    print "------------------------"
    print "BR: H(", 100*h,"%), Z(", 100*z,"%)"
    print ("Decay composition: HH %.2f, HZ %.2f, ZZ %.2f" % (h*h*100, 2*h*z*100, z*z*100))
