# Module imports
import skimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
import warnings
warnings.filterwarnings("ignore")

# Question 1
img = skimage.io.imread("./data/moon_noisy.png")


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

def adaptive_binarization(img, block_size):
    M, N = img.shape
    m, n = M//block_size, N//block_size
    v_s, h_s = 0, 0
    binarized_image = np.zeros((M, N))
    t, min_within_class_variance = None, None

    def within_class_variance(t, p):  # sw^2 = w0 * s0^2 + w1 * s1^2
        # Class probabilities
        w0 = np.sum(p[:t+1])
        w1 = np.sum(p[t+1:])
        if w0==0 or w1==0:
            return np.inf
        # Mean value of intensity of a pixel belonging to a class
        u0 = np.sum([k*p[k] for k in range(t+1)]) / w0
        u1 = np.sum([k*p[k] for k in range(t+1, 256)]) / w1
        # Variance of intensities of pixel belonging to a class
        s0 = np.sum([((k-u0)**2)*p[k] for k in range(t+1)]) / w0
        s1 = np.sum([((k-u1)**2)*p[k] for k in range(t+1, 256)]) / w1
        # Within class variance
        return w0*s0 + w1*s1

    while v_s < block_size:
        freq = np.zeros(256)
        img_plot = img[h_s*m:h_s*m+n, v_s*n:v_s*n+m]
        for i in range(256):
            freq[i] = np.sum(img_plot==i)
        p = freq / (m*n)
        within_class_variance_t = np.zeros(256)
        for t in range(256):
            within_class_variance_t[t] = within_class_variance(t, p)
        t = np.argmin(within_class_variance_t)
        min_within_class_variance = np.min(within_class_variance_t)
        for i in range(m):
            for j in range(n):
                if img[h_s*m+i, v_s*n+j] > t:
                    binarized_image[h_s*m+i,v_s*n+j] = 1
                else:
                    binarized_image[h_s*m+i,v_s*n+j] = 0
        h_s += 1
        if h_s == block_size:
            h_s = 0
            v_s += 1
    return min_within_class_variance, t, binarized_image

def kernel(M, N, s_g):
    filter = np.zeros((M, N))
    if s_g==0:
        filter[M//2, N//2] = 1
        return filter
    for i in np.arange(-M//2, M//2+1, 1):
        for j in np.arange(-N//2, N//2+1, 1):
            filter[i+M//2, j+N//2] = np.exp(-((i**2+j**2)/(2*s_g**2)))
    filter = filter / np.sum(filter)
    return filter

s_gs = np.array([0, 0.1, 0.5, 1, 2.5, 5, 10, 20])
within_class_variance = np.zeros(len(s_gs))
fig, axs = plt.subplots(8, 3, figsize=(16, 52))
axes = axs.flat
for i in range(len(s_gs)):
    filter = kernel(41, 41, s_gs[i])
    img_ = img.copy()
    img_[:,:,0] = cv2.filter2D(img[:,:,0],0, filter)
    img_[:,:,1] = cv2.filter2D(img[:,:,1],0, filter)
    img_[:,:,2] = cv2.filter2D(img[:,:,2],0, filter)
    axes[3*i].imshow(img_,)
    axes[3*i].set_title(f"$ \sigma_g$ = {s_gs[i]}")
    grayscale_img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    within_class_variance[i], t, binarized_img = adaptive_binarization(grayscale_img, block_size=1)
    axes[3*i+1].imshow(binarized_img, 'gray')
    axes[3*i+1].set_title(f"$ \sigma_g$ = {s_gs[i]} t = {t} $ \sigma_w^2$ =  {np.round(within_class_variance[i], 2)}")

    freq = np.zeros(256)
    for j in range(256):
        freq[j] = np.sum(grayscale_img==j)

    axes[3*i+2].plot(freq)
    axes[3*i+2].axvline(t, color="black", linestyle="--", label="t*")
    axes[3*i+2].set_xlabel("Intensity")
    axes[3*i+2].set_ylabel("Frequency")
    axes[3*i+2].set_title(f"t = {t}")
print(f's_g that minimizes s_w**2: {s_gs[np.argmin(within_class_variance)]}')
plt.tight_layout()
plt.savefig('1a.png')
plt.show()

