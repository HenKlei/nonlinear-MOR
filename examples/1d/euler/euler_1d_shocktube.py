import numpy as np

from clawpack import pyclaw
from clawpack import riemann
from clawpack.riemann.euler_with_efix_1D_constants import density, momentum, energy, num_eqn
from clawpack.pyclaw.examples.euler_1d.shocktube import setplot

from nonlinear_mor.models import WrappedPyClawModel
from nonlinear_mor.utils.parameters import CubicParameterSpace


def create_model(spatial_shape, num_time_steps, spatial_extend=[(0., 1.), (0., 1.)], t_final=0.6):
    def call_pyclaw(claw, domain, mu):
        riemann_solver = riemann.euler_1D_py.euler_hllc_1D
        claw.solver = pyclaw.ClawSolver1D(riemann_solver)
        claw.solver.kernel_language = 'Python'

        claw.solver.bc_lower[0] = pyclaw.BC.extrap
        claw.solver.bc_upper[0] = pyclaw.BC.extrap

        mx = spatial_shape[0]
        x = pyclaw.Dimension(-1.0, 1.0, mx, name='x')
        domain = pyclaw.Domain([x])
        state = pyclaw.State(domain, num_eqn)

        state.problem_data['gamma'] = mu
        state.problem_data['gamma1'] = mu - 1.

        x = state.grid.x.centers

        rho_l = 1.
        rho_r = 1./8
        p_l = 1.
        p_r = 0.1
        state.q[density, :] = (x < 0.)*rho_l + (x >= 0.)*rho_r
        state.q[momentum, :] = 0.
        velocity = state.q[momentum, :]/state.q[density, :]
        pressure = (x < 0.)*p_l + (x >= 0.)*p_r
        state.q[energy, :] = pressure / (mu - 1.) + 0.5 * state.q[density, :] * velocity**2

        claw.tfinal = 0.4
        claw.solution = pyclaw.Solution(state, domain)
        claw.num_output_times = num_time_steps - 1
        claw.setplot = setplot
        claw.keep_copy = True  # keep solution data in memory for returning
        claw.output_format = None  # do not write solution data to file

        claw.run()

        u = np.array([s.q[0] for s in claw.frames])
        return u

    parameter_space = CubicParameterSpace([(1.2, 3)])

    return WrappedPyClawModel(spatial_shape, num_time_steps, parameter_space, spatial_extend, t_final, call_pyclaw)
