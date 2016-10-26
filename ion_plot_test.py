import numpy as np
import matplotlib.pyplot as plt


for i in range(10):
    plt.clf()
    plt.axis([0, 10, 0, 1])
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(1.0)


while True:
    plt.pause(0.05)


plt.ion()