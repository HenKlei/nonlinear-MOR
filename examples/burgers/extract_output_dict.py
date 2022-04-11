import pickle

filepath_prefix = 'siamuq_results'

with open(filepath_prefix + '/outputs/output_dict_rom', 'rb') as output_file:
    output_dict = pickle.load(output_file)

with open(filepath_prefix + '/outputs/full_velocity_fields', 'wb') as output_file:
    pickle.dump(output_dict['full_velocity_fields'], output_file)
