import matplotlib.pyplot as plt

from shallow_water_dam_break import create_model


model = create_model((100, ), 50, spatial_extent=[(-5., 5.)], t_final=2.)
u = model.solve(0.25)
ani = model.visualize(u)
u.plot(extent=(-5., 5., 0., 2.))
plt.show()
