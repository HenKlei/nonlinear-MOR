import numpy as np

from clawpack import pyclaw
from clawpack import riemann
from clawpack.pyclaw.examples.acoustics_2d_variable.acoustics_2d_interface import setplot

from nonlinear_mor.models import WrappedPyClawModel
from nonlinear_mor.utils.parameters import CubicParameterSpace


def create_model(spatial_shape, num_time_steps, spatial_extend=[(-1., 1.), (-1., 1.)], t_final=0.6):
    def call_pyclaw(claw, domain, mu):
        mu = mu[0]
        claw.solver = pyclaw.ClawSolver2D(riemann.vc_acoustics_2D)
        claw.solver.dimensional_split = False
        claw.solver.limiters = pyclaw.limiters.tvd.MC

        claw.solver.bc_lower[0] = pyclaw.BC.wall
        claw.solver.bc_upper[0] = pyclaw.BC.extrap
        claw.solver.bc_lower[1] = pyclaw.BC.wall
        claw.solver.bc_upper[1] = pyclaw.BC.extrap
        claw.solver.aux_bc_lower[0] = pyclaw.BC.wall
        claw.solver.aux_bc_upper[0] = pyclaw.BC.extrap
        claw.solver.aux_bc_lower[1] = pyclaw.BC.wall
        claw.solver.aux_bc_upper[1] = pyclaw.BC.extrap

        x = pyclaw.Dimension(spatial_extend[0][0], spatial_extend[0][1], spatial_shape[0], name='x')
        y = pyclaw.Dimension(spatial_extend[1][0], spatial_extend[1][1], spatial_shape[1], name='y')
        domain = pyclaw.Domain([x, y])

        num_eqn = 3
        num_aux = 2  # density, sound speed
        state = pyclaw.State(domain, num_eqn, num_aux)

        grid = state.grid
        X, Y = grid.p_centers

        rho_left = 4.0  # Density in left half
        rho_right = mu  # Density in right half
        bulk_left = 4.0  # Bulk modulus in left half
        bulk_right = 4.0  # Bulk modulus in right half
        c_left = np.sqrt(bulk_left / rho_left)  # Sound speed (left)
        c_right = np.sqrt(bulk_right / rho_right)  # Sound speed (right)
        state.aux[0, :, :] = rho_left * (X < 0.) + rho_right * (X >= 0.)  # Density
        state.aux[1, :, :] = c_left * (X < 0.) + c_right * (X >= 0.)  # Sound speed

        # Set initial condition
        x0 = -0.5
        y0 = 0.
        r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        width = 0.1
        rad = 0.25
        state.q[0, :, :] = (np.abs(r - rad) <= width) * (1. + np.cos(np.pi * (r - rad) / width))
        state.q[1, :, :] = 0.
        state.q[2, :, :] = 0.

        claw.keep_copy = True
        claw.output_format = None
        claw.solution = pyclaw.Solution(state, domain)
        claw.tfinal = t_final
        claw.num_output_times = num_time_steps - 1
        claw.write_aux_init = True
        claw.setplot = setplot

        claw.run()

        u = np.array([s.q[0].T for s in claw.frames])
        return u

    parameter_space = CubicParameterSpace([(1., 10.)])

    return WrappedPyClawModel(spatial_shape, num_time_steps, parameter_space, spatial_extend, t_final, call_pyclaw,
                              name='2dAcousticsPyClawVariableCoefficients')
