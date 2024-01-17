import numpy as np

from clawpack import pyclaw
from clawpack import riemann
from clawpack.riemann.shallow_roe_with_efix_1D_constants import depth, momentum, num_eqn
from clawpack.pyclaw.examples.shallow_1d.dam_break import setplot

from nonlinear_mor.models import WrappedPyClawModel
from nonlinear_mor.utils.parameters import CubicParameterSpace


def create_model(spatial_shape, num_time_steps, spatial_extent=[(-5., 5.)], t_final=2.0):
    def call_pyclaw(claw, domain, mu):
        riemann_solver = riemann.shallow_1D_py.shallow_hll_1D
        claw.solver = pyclaw.ClawSolver1D(riemann_solver)
        claw.solver.limiters = pyclaw.limiters.tvd.vanleer
        claw.solver.kernel_language = 'Python'

        claw.solver.bc_lower[0] = pyclaw.BC.extrap
        claw.solver.bc_upper[0] = pyclaw.BC.extrap

        xlower = spatial_extent[0][0]
        xupper = spatial_extent[0][1]
        mx = spatial_shape[0]
        x = pyclaw.Dimension(xlower, xupper, mx, name='x')
        domain = pyclaw.Domain(x)
        state = pyclaw.State(domain, num_eqn)

        # Gravitational constant as parameter
        state.problem_data['grav'] = mu[0]
        state.problem_data['dry_tolerance'] = 1e-3
        state.problem_data['sea_level'] = 0.0

        xc = state.grid.x.centers

        x0 = 0.

        hl = 3.
        ul = 0.
        hr = 1.
        ur = 0.
        state.q[depth, :] = hl * (xc <= x0) + hr * (xc > x0)
        state.q[momentum, :] = hl * ul * (xc <= x0) + hr * ur * (xc > x0)

        claw.tfinal = t_final
        claw.solution = pyclaw.Solution(state, domain)
        claw.num_output_times = num_time_steps - 1
        claw.setplot = setplot
        claw.keep_copy = True  # keep solution data in memory for returning
        claw.output_format = None  # do not write solution data to file

        claw.run()

        u = np.array([s.q[0] for s in claw.frames])
        return u

    parameter_space = CubicParameterSpace([(0.5, 1.5)])

    return WrappedPyClawModel(spatial_shape, num_time_steps, parameter_space, spatial_extent, t_final, call_pyclaw,
                              name='1dShallowWaterDamBreakPyClaw')
