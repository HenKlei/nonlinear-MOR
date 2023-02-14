import matplotlib.pyplot as plt
import numpy as np

from burgers_1d_pymor_two_parameters import create_model

model = create_model((100, ), 200)
u = model.solve(np.array([1, 2]))
model.visualize(u)
u.plot()
plt.show()
