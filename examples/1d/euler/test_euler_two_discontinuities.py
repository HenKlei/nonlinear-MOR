import matplotlib.pyplot as plt

from euler_shocktube_two_discontinuities import create_model


model = create_model((100, ), 50, spatial_extent=[(-1., 1.)], t_final=0.4)
u = model.solve(1.4)
ani = model.visualize(u)
u.plot(extent=(-1., 1., 0., 0.4))
plt.show()
