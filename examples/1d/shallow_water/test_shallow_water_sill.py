import matplotlib.pyplot as plt

from shallow_water_sill import create_model


model = create_model((100, ), 50, spatial_extent=[(-1., 1.)], t_final=1.)
u = model.solve(9.8)
ani = model.visualize(u)
u.plot(extent=(-1., 1., 0., 1.))
plt.show()
