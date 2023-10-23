import matplotlib.pyplot as plt

from burgers import create_model

model = create_model((100, 50), 200)
u = model.solve(1)
model.visualize(u)
plt.show()
