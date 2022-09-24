import numpy as np
import matplotlib.pyplot as plt

gm = np.load("goodMoves0.npy" , allow_pickle=True)

for moves in gm:
    print(moves[1], moves[2])
    plt.imshow(moves[0])
    plt.show()
    