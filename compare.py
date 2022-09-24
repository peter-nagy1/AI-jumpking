import sys
from scipy.linalg import norm
from scipy import sum, average
import numpy as np
import cv2
import matplotlib.pyplot as plt

stage1 = np.load("level1_1.npy")


plt.imshow(stage1[316])
plt.figure()
plt.imshow(stage1[325])

plt.show()

def compare_images(img1, img2):

    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = np.sum(abs(diff))  # Manhattan norm

    return m_norm

""" norms = []
i_s = []
j_s = []
for i in range(len(stage1[:500])):
    for j in range(len(stage1[:500])):
        if i != j:
            print(i, j)
            norm = compare_images(stage1[i], stage1[j])
            
            if norm < 50:
                norms.append(norm)
                i_s.append(i)
                j_s.append(j)

"""