import numpy as np

from clawpack import pyclaw
from clawpack import riemann

from nonlinear_mor.models import WrappedPyClawModel
from nonlinear_mor.utils.parameters import CubicParameterSpace


def create_model(spatial_shape, num_time_steps, spatial_extend=[(0., 1.), (0., 1.)], t_final=0.6):
    def call_pyclaw(claw, domain, mu):
        riemann_solver = riemann.euler_4wave_2D
        claw.solver = pyclaw.ClawSolver2D(riemann_solver)
        claw.solver.all_bcs = pyclaw.BC.extrap

        claw.solution = pyclaw.Solution(claw.solver.num_eqn, domain)
        claw.solution.problem_data['gamma'] = mu

        # Set initial data
        q = claw.solution.q
        xx, yy = domain.grid.p_centers
        l = (xx < 0.5)
        r = (xx >= 0.5)
        b = (yy < 0.5)
        t = (yy >= 0.5)
        q[0] = 2. * l * t + 1. * l * b + 1. * r * t + 3. * r * b
        q[1] = 0.75 * t - 0.75 * b
        q[2] = 0.5 * l - 0.5 * r
        q[3] = 0.5 * q[0] * (q[1]**2 + q[2]**2) + 1. / (mu - 1.)

        claw.keep_copy = True  # keep solution data in memory for returning
        claw.output_format = None  # do not write solution data to file
        claw.solver.dt_initial = 1.e99

        claw.run()

        u = np.array([s.q[3] for s in claw.frames])
        return u

    parameter_space = CubicParameterSpace([(1.2, 3)])

    return WrappedPyClawModel(spatial_shape, num_time_steps, parameter_space, spatial_extend, t_final, call_pyclaw)
