import matplotlib.pyplot as plt
import numpy as np

from burgers_two_parameters_central_merging import create_model

model = create_model((100, ), 100)
u = model.solve(np.array([2., 0.5]))
model.visualize(u)
u.plot()
plt.show()
