from typer import Argument, run
import pickle


def main(filepath_prefix: str = Argument(..., help='Prefix of the filepath to load the output dictionary from ' +
                                                   'and write the velocity fields file to.')):
    with open(filepath_prefix + '/outputs/output_dict_rom', 'rb') as output_file:
        output_dict = pickle.load(output_file)

    with open(filepath_prefix + '/outputs/full_velocity_fields', 'wb') as output_file:
        pickle.dump(output_dict['full_velocity_fields'], output_file)


if __name__ == "__main__":
    run(main)
