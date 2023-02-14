import matplotlib.pyplot as plt

from euler_1d_shocktube import create_model


model = create_model((100, ), 50)
u = model.solve(1.4)
ani = model.visualize(u)
u.plot()
plt.show()
