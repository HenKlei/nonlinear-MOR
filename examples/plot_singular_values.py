import numpy as np
import matplotlib.pyplot as plt
from typer import Argument, run


def main(path: str = Argument(..., help='Path to folder in which to find the singular value files')):
    singular_values_vector_fields = np.loadtxt(path + "singular_values.txt")
    singular_values_snapshots = np.loadtxt(path + "singular_values_snapshots.txt")

    singular_values_vector_fields /= singular_values_vector_fields[0]
    singular_values_snapshots /= singular_values_snapshots[0]

    fig, axis = plt.subplots()
    axis.semilogy(singular_values_vector_fields, label="Vector fields")
    axis.semilogy(singular_values_snapshots, label="Snapshots")
    fig.legend()
    fig.suptitle("Singular values")
    plt.show()


if __name__ == "__main__":
    run(main)
