import pickle
import matplotlib.pyplot as plt

from geodesic_shooting.utils.visualization import plot_vector_field

from nonlinear_mor.utils.torch.early_stopping import SimpleEarlyStoppingScheduler
from nonlinear_mor.utils.torch.neural_networks import FullyConnectedNetwork
from nonlinear_mor.utils.torch.trainer import Trainer


with open('outputs/output_dict_rom', 'rb') as output_file:
    output_dict = pickle.load(output_file)

print(output_dict.keys())

print(output_dict['reduced_velocity_fields'].shape)
v0 = output_dict['reduced_velocity_fields'].reshape((2, 100, 100))
plot_vector_field(v0, title="Reduced initial vector field (C2S)", interval=2)
plt.show()


validation_data = output_dict['validation_data']
training_data = output_dict['training_data']
layers_sizes = [1, 30, 30, 1]

best_neural_network = None
best_loss = None

restarts = 100


def train_neural_network(layers_sizes, training_data, validation_data):
    neural_network = FullyConnectedNetwork(layers_sizes)
    trainer = Trainer(neural_network, es_scheduler=SimpleEarlyStoppingScheduler)
    best_loss, _ = trainer.train(training_data, validation_data)
    return trainer.network, best_loss


print(f"Performing {restarts} restarts of neural network training ...")
for _ in range(restarts):
    neural_network, loss = train_neural_network(layers_sizes, training_data, validation_data)
    if best_loss is None or best_loss > loss:
        best_neural_network = neural_network
        best_loss = loss

print(f"Trained neural network with loss of {best_loss} ...")
