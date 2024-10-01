import itertools

example = ['1d.burgers.piecewise_constant.burgers_landmarks_analytical']
spatial_shape = [100]  # This could also be increased substantially to show the independence of the computational effort from the resolution (becomes only relevant for reconstruction of solution)
num_time_steps = [100]  # Same as for the spatial shape!
additional_parameters = [{}]
place_landmarks_automatically = ['--no-place-landmarks-automatically']
num_landmarks = [20]
landmarks_labeled = ['--landmarks-labeled', '--no-landmarks-labeled']
num_training_parameters = [20]
num_test_parameters = [50]
sampling_mode = ['uniform']
oversampling_size = [0]
reference_parameter = [0.5, 1.0, 1.5]
sigma = [0.01, 0.1]
kernel_sigma = [8., 4., 1.]
kernel_dist_sigma = [-1]
write_results = ["--write-results"]

with open("run_commands_in_screens.txt", "w") as f:
    for vals in itertools.product(example, spatial_shape, num_time_steps, additional_parameters, place_landmarks_automatically, num_landmarks, landmarks_labeled, num_training_parameters, num_test_parameters, sampling_mode, oversampling_size, reference_parameter, sigma, kernel_sigma, kernel_dist_sigma, write_results):
        e, nx, nt, ap, lm_auto, num_lm, lm_ld, num_train_p, num_test_p, sm, os, rp, si, k_si, k_dist_si, wr = vals
        f.write(f"source ../../venv/bin/activate && python landmarks_registration_tests.py {e} {nx} --num-time-steps {nt} --place-landmarks-automatically {lm_auto} --num-landmarks {num_lm} --landmarks-labeled {lm_ld} --num-training-parameters {num_train_p} --num-test-parameters {num_test_p} --sampling-mode {sm} --oversampling-size {os} --reference-parameter {rp} --sigma {si} --kernel-sigma {k_si} --kernel-dist-sigma {k_dist_si} --write-results {wr}\n")
