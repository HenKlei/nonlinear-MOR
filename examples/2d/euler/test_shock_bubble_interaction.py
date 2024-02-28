import matplotlib.pyplot as plt

from shock_bubble_interaction import create_model

model = create_model((160, 40), 20)
u = model.solve(1.6)
ani = model.visualize(u)
plt.show()
