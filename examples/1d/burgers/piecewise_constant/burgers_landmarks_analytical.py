import numpy as np

from .burgers_analytical import create_model  # noqa: F401


def get_landmarks(mu=0.25, all_landmarks=True, many_landmarks=False):
    t_central_intersection = 0.25 / mu
    x_central_intersection = 0.625
    x_boundary_intersection = 1.
    t_boundary_intersection = 0.625 / mu
    if t_boundary_intersection > 1.:
        x_boundary_intersection = mu + 0.375
        t_boundary_intersection = 1.
    if all_landmarks:
        if many_landmarks:
            return np.array([[x_central_intersection, t_central_intersection],
                             [x_boundary_intersection, t_boundary_intersection],
                             [x_central_intersection + 0.0625 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.0625 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.125 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.125 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.1875 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.1875 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.25 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.25 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.3125 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.3125 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.375 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.375 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.4375 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.4375 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.5 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.5 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.5625 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.5625 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.625 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.625 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.6875 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.6875 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.75 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.75 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.8125 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.8125 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.875 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.875 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.9375 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.9375 * (t_boundary_intersection - t_central_intersection)],
                             [0.25, 0.],
                             [0.25 + 0.125 * (x_central_intersection - 0.25), 0.125 * t_central_intersection],
                             [0.25 + 0.25 * (x_central_intersection - 0.25), 0.25 * t_central_intersection],
                             [0.25 + 0.375 * (x_central_intersection - 0.25), 0.375 * t_central_intersection],
                             [0.25 + 0.5 * (x_central_intersection - 0.25), 0.5 * t_central_intersection],
                             [0.25 + 0.625 * (x_central_intersection - 0.25), 0.625 * t_central_intersection],
                             [0.25 + 0.75 * (x_central_intersection - 0.25), 0.75 * t_central_intersection],
                             [0.25 + 0.875 * (x_central_intersection - 0.25), 0.875 * t_central_intersection],
                             [0.5 + 0.125 * (x_central_intersection - 0.5), 0.125 * t_central_intersection],
                             [0.5 + 0.25 * (x_central_intersection - 0.5), 0.25 * t_central_intersection],
                             [0.5 + 0.375 * (x_central_intersection - 0.5), 0.375 * t_central_intersection],
                             [0.5 + 0.5 * (x_central_intersection - 0.5), 0.5 * t_central_intersection],
                             [0.5 + 0.625 * (x_central_intersection - 0.5), 0.625 * t_central_intersection],
                             [0.5 + 0.75 * (x_central_intersection - 0.5), 0.75 * t_central_intersection],
                             [0.5 + 0.875 * (x_central_intersection - 0.5), 0.875 * t_central_intersection],
                             [0.5, 0.]])
        else:
            return np.array([[x_central_intersection, t_central_intersection],
                             [x_boundary_intersection, t_boundary_intersection],
                             [x_central_intersection + 0.125 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.125 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.25 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.25 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.375 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.375 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.5 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.5 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.625 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.625 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.75 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.75 * (t_boundary_intersection - t_central_intersection)],
                             [x_central_intersection + 0.875 * (x_boundary_intersection - x_central_intersection),
                              t_central_intersection + 0.875 * (t_boundary_intersection - t_central_intersection)],
                             [0.25, 0.],
                             [0.25 + 0.25 * (x_central_intersection - 0.25), 0.25 * t_central_intersection],
                             [0.25 + 0.5 * (x_central_intersection - 0.25), 0.5 * t_central_intersection],
                             [0.25 + 0.75 * (x_central_intersection - 0.25), 0.75 * t_central_intersection],
                             [0.5 + 0.25 * (x_central_intersection - 0.5), 0.25 * t_central_intersection],
                             [0.5 + 0.5 * (x_central_intersection - 0.5), 0.5 * t_central_intersection],
                             [0.5 + 0.75 * (x_central_intersection - 0.5), 0.75 * t_central_intersection],
                             [0.5, 0.]])
    else:
        return np.array([[x_central_intersection, t_central_intersection],
                         [x_boundary_intersection, t_boundary_intersection]])
