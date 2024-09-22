import skimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def between_class_variance(img, t): 
    freq = np.zeros(256)
    for i in range(256):
        freq[i] = np.sum(img==i)
    m, n = img.shape
    p = freq/(m*n)
    # Class probabilities
    w0 = np.sum(p[:t+1])
    w1 = np.sum(p[t+1:])
    # image mean
    uT = np.sum(np.arange(256)*p)
    if w0==0 or w1==0:
        return -np.inf
    # Mean value of intensity of a pixel belonging to a class
    u0 = np.sum(np.arange(t+1)*p[:t+1])/w0
    u1 = np.sum(np.arange(t+1,256)*p[t+1:])/w1
    # Between class variance
    return w0*(u0-uT)**2 + w1*(u1-uT)**2 

def kernel(x, y, s_g):
    return np.exp(-((x**2+y**2)/(2*s_g**2)))

for i in arnag