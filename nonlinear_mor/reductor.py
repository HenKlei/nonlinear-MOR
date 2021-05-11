from tent_pitching.functions import SpaceTimeFunction
from tent_pitching.grids import SpaceTimeGrid


class NonlinearReductor:
    def __init__(self, space_time_grid):
        assert isinstance(space_time_grid, SpaceTimeGrid)
        self.space_time_grid = space_time_grid

    def reduce(self, solutions, tol=1e-2, max_basis_size=10):
        assert isinstance(solutions, list)
        assert all(isinstance(solution, tuple) and len(solution) == 2
                   and isinstance(solution[1], SpaceTimeFunction)
                   for solution in solutions)
        assert isinstance(tol, float) and tol > 0
        assert isinstance(max_basis_size, int) and max_basis_size > 0

        reduced_basis = []
        velocity_fields = []

        for tent in self.space_time_grid.tents:
            local_reduced_basis = []
            local_velocity_fields = []
            local_solutions = [(solution[0], solution[1].get_function_on_tent(tent))
                               for solution in solutions]

            iterations = 0
            while iterations <= max_basis_size:
                max_error = None
                local_solution_with_largest_error = None
                for _, local_solution in local_solutions:
                    error = self.compute_local_error(local_solution, local_reduced_basis,
                                                     local_velocity_fields)
                    if max_error is None or error > max_error:
                        max_error = error
                        local_solution_with_largest_error = local_solution

                iterations += 1

                local_reduced_basis.append(local_solution_with_largest_error)
                local_velocity_fields.append() # local_velocity_fields = ....

            reduced_basis.append(local_reduced_basis)
            velocity_fields.append(local_velocity_fields)

        return reduced_basis, velocity_fields

    def compute_local_error(self, local_solution, local_reduced_basis, local_velocity_fields):

        raise NotImplementedError
