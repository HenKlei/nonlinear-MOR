import matplotlib.pyplot as plt
import numpy as np

from burgers import create_model


model = create_model((256, 128), 127)
u = model.solve(1)
with open("sol_mu_1.npy", "wb") as f:
    np.save(f, u.to_numpy()[:128])
model.visualize(u)
plt.show()
u = model.solve(1.5)
with open("sol_mu_2.npy", "wb") as f:
    np.save(f, u.to_numpy()[:128])
model.visualize(u)
plt.show()
