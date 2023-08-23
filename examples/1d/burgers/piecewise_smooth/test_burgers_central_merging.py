import matplotlib.pyplot as plt

from burgers_1d_pymor_central_merging import create_model

model = create_model((100, ), 100)
u = model.solve(1.75)
model.visualize(u)
u.plot()
plt.show()
