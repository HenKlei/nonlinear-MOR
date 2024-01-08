import numpy as np

from clawpack import pyclaw
from clawpack import riemann
from clawpack.pyclaw.examples.shallow_1d.sill import setplot

from nonlinear_mor.models import WrappedPyClawModel
from nonlinear_mor.utils.parameters import CubicParameterSpace


def create_model(spatial_shape, num_time_steps, spatial_extent=[(-1., 1.)], t_final=1.0):
    def call_pyclaw(claw, domain, mu):
        riemann_solver = riemann.shallow_1D_py.shallow_fwave_1d
        claw.solver = pyclaw.ClawSolver1D(riemann_solver)
        claw.solver.limiters = pyclaw.limiters.tvd.vanleer
        claw.solver.kernel_language = 'Python'

        claw.solver.fwave = True
        claw.solver.num_waves = 2
        claw.solver.num_eqn = 2
        claw.solver.bc_lower[0] = pyclaw.BC.extrap
        claw.solver.bc_upper[0] = pyclaw.BC.extrap
        claw.solver.aux_bc_lower[0] = pyclaw.BC.extrap
        claw.solver.aux_bc_upper[0] = pyclaw.BC.extrap

        xlower = spatial_extent[0][0]
        xupper = spatial_extent[0][1]
        mx = spatial_shape[0]
        x = pyclaw.Dimension(xlower, xupper, mx, name='x')
        domain = pyclaw.Domain(x)
        state = pyclaw.State(domain, 2, 1)

        # Gravitational constant
        state.problem_data['grav'] = mu
        state.problem_data['dry_tolerance'] = 1e-3
        state.problem_data['sea_level'] = 0.0

        xc = state.grid.x.centers
        state.aux[0, :] = 0.8 * np.exp(-xc ** 2 / 0.2 ** 2) - 1.0
        state.q[0, :] = 0.1 * np.exp(-(xc + 0.4) ** 2 / 0.2 ** 2) - state.aux[0, :]
        state.q[1, :] = 0.0

        claw.tfinal = t_final
        claw.solution = pyclaw.Solution(state, domain)
        claw.num_output_times = num_time_steps - 1
        claw.setplot = setplot
        claw.keep_copy = True  # keep solution data in memory for returning
        claw.output_format = None  # do not write solution data to file

        claw.run()

        u = np.array([s.q[0] for s in claw.frames])
        return u

    parameter_space = CubicParameterSpace([(5., 10.)])

    return WrappedPyClawModel(spatial_shape, num_time_steps, parameter_space, spatial_extent, t_final, call_pyclaw,
                              name='1dShallowWaterSillPyClaw')
