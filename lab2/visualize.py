import matplotlib.pyplot as plt
import numpy as np
accuracy = []
with open('means.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        accuracy.append(float(line))

plt.plot(np.arange(len(accuracy))*1000, accuracy, color='brown', linestyle="-", markersize="16", label="accuracy")
plt.xlabel("episode")
plt.ylabel("score")
plt.legend()
plt.savefig('accuracy.png')
