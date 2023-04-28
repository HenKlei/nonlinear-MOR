import itertools

example = ['1d.burgers.piecewise_constant.burgers_1d_analytical']
spatial_shape = [100]
num_time_steps = [100]
additional_parameters = [{}]
num_training_parameters = [20]
sampling_mode = ['uniform']
reference_parameter = [0.625]
alpha = [0.1, 0.01]
exponent = [1]
gamma = [1.]
sigma = [0.01]
oversampling_size = [10]
optimization_method = ['L-BFGS-B']
max_reduced_basis_size = [20]
num_workers = [1]
l2_prod = ["--no-l2-prod"]
neural_network_training_restarts = [10]
#hidden_layers = [[20, 20, 20]]
interval = [1]
#full_vector_fields_filepath_prefix = [None]
write_results = ["--write-results"]

with open("run_commands_in_screens.txt", "w") as f:
    for vals in itertools.product(example, spatial_shape, num_time_steps, additional_parameters,
                                  num_training_parameters, sampling_mode, reference_parameter, alpha, exponent, gamma,
                                  sigma, oversampling_size, optimization_method, max_reduced_basis_size, num_workers,
                                  l2_prod, neural_network_training_restarts,
                                  #hidden_layers,
                                  interval,
                                  #full_vector_fields_filepath_prefix,
                                  write_results):
#        e, nx, nt, ap, n_train, sm, rp, al, exp, gam, si, os, opm, max_rb, num_w, pr, nntr, hl, i, fvvf, wr = vals
        e, nx, nt, ap, n_train, sm, rp, al, exp, gam, si, os, opm, max_rb, num_w, pr, nntr, i, wr = vals
        f.write(f"source ../../venv/bin/activate && python perform_reduction.py {e} {nx} --num-time-steps {nt} "
                f"--additional-parameters {ap} --num-training-parameters {n_train} --sampling-mode {sm} "
                f"--reference-parameter {rp} --alpha {al} --exponent {exp} --gamma {gam} --sigma {si} "
                f"--oversampling-size {os} --optimization-method {opm} --max-reduced-basis-size {max_rb} "
                f"--num-workers {num_w} {pr} --neural-network-training-restarts {nntr} "
#                f"--hidden-layers {hl} "
                f"--interval {i} "
#                f"--full-vector-fields-filepath-prefix {fvvf} "
                f"{wr}\n")
