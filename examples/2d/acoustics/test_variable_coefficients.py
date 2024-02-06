import matplotlib.pyplot as plt

from variable_coefficients import create_model

model = create_model((100, 100), 20)
u = model.solve(1.4)
ani = model.visualize(u)
plt.show()
