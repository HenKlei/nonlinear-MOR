import itertools

example = ['1d.shallow_water.shallow_water_dam_break']
spatial_shape = [100]
num_time_steps = [100]
additional_parameters = [{}]
num_training_parameters = [20]
sampling_mode = ['uniform']
reference_parameters = ["[[0.5], [1.]]"]
alpha = [0.1, 0.01]
exponent = [1]
gamma = [1., 5.]
sigma = [0.01, 0.1, 0.05]
oversampling_size = [0]
optimization_method = ['L-BFGS-B']
l2_prod = ["--no-l2-prod"]
reuse_initial_vector_field = ["--no-reuse-initial-vector-field"]
write_results = ["--write-results"]

with open("run_commands_in_screens.txt", "w") as f:
    for vals in itertools.product(example, spatial_shape, num_time_steps, additional_parameters,
                                  num_training_parameters, sampling_mode, reference_parameters, alpha, exponent, gamma,
                                  sigma, oversampling_size, optimization_method, l2_prod, reuse_initial_vector_field,
                                  write_results):
        e, nx, nt, ap, n_train, sm, rp, al, exp, gam, si, os, opm, pr, ri, wr = vals
        f.write(f"source ../../venv/bin/activate && python test_registration.py {e} {nx} --num-time-steps {nt} "
                f"--additional-parameters {ap} --num-training-parameters {n_train} --sampling-mode {sm} "
                f"--reference-parameters {rp} --alpha {al} --exponent {exp} --gamma {gam} --sigma {si} "
                f"--oversampling-size {os} --optimization-method {opm} {pr} {ri} {wr}\n")
