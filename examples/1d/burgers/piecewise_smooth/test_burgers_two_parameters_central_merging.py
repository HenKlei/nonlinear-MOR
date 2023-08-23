import matplotlib.pyplot as plt
import numpy as np

from burgers_1d_pymor import create_model

model = create_model((100, ), 100)
u = model.solve(np.array([2.0, 0.5]))
model.visualize(u)
u.plot()
plt.show()
