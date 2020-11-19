import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 180 * 4)
y = np.cos(3/30 * np.pi * t) + np.cos(2/34 * np.pi * t)

plt.plot(t, y)
plt.show()
