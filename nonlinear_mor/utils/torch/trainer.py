import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

import numpy as np

from .dataset import CustomDataset
from .progressbar import ProgressTraining

from nonlinear_mor.utils.logger import getLogger
from nonlinear_mor.utils.torch.early_stopping import SimpleEarlyStoppingScheduler


class Trainer:
    """Class that implements a generic trainer for neural networks.

    Parameters
    ----------
    optimizer
        Optimizer to use for training.
    parameter_optimizer
        Additional parameters for the optimizer.
    learning_rate
        Initial default learning rate for the optimizer.
    lr_scheduler
        Learning rate scheduler to use.
    parameters_lr_scheduler
        Additional parameters for the learning rate scheduler.
    es_scheduler
        Early stopping scheduler to use.
    parameters_es_scheduler
        Additional parameters for the early stopping scheduler.
    """
    def __init__(self, network, optimizer=optim.LBFGS, parameters_optimizer={}, learning_rate=1.,
                 loss_function=nn.MSELoss(), lr_scheduler=None, parameters_lr_scheduler={},
                 es_scheduler=SimpleEarlyStoppingScheduler, parameters_es_scheduler={}):
        self.network = network

        self.optimizer = optimizer(self.network.parameters(), lr=learning_rate,
                                   **parameters_optimizer)
        self.learning_rate = learning_rate
        self.loss_function = loss_function

        if lr_scheduler:
            lr_scheduler = lr_scheduler(self.optimizer, **parameters_lr_scheduler)
        self.lr_scheduler = lr_scheduler
        if es_scheduler:
            es_scheduler = es_scheduler(self, **parameters_es_scheduler)
        self.es_scheduler = es_scheduler

        self.logger = getLogger('nonlinear_mor.Trainer')

    def train(self, training_data, validation_data, number_of_training_samples=100,
              number_of_epochs=1000, batch_size=25, learning_rate=None,
              number_of_validation_samples=20, show_progress_bar=True):
        """Sets up everything and call function to start training procedure.

        Parameters
        ----------
        training_data
            Already computed training data (if available).
        training_loader
            Data loader holding the training data and performing
            the (random) mini-batching of the training data.
        number_of_training_samples
            If no training data is provided, this determines the number of
            training samples to compute using the respective function of the
            model.
        number_of_epochs
            Maximum number of training epochs to perform.
        batch_size
            Batch size to use for mini-batching (has to be the size of the
            training set when using L-BFGS as optimizer).
        learning_rate
            Initial learning rate for the each of the parameter groups of the
            optimizer.
        validation_data
            Already compute validation data (if available).
        validation_loader
            Data loader holding the validation data and performing
            the (random) mini-batching of the validation data.
        number_of_validation_samples
            If no validation data is provided, this determines the number of
            validation samples to compute using the respective function of the
            model.
        show_progress_bar
            Determines whether or not to show a progress bar during training.

        Returns
        -------
        Minimum validation and training loss (if early stopping
        scheduler is used).
        """
        # set learning rate for each parameter group in the optimizer to the given initial value
        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = learning_rate

        self.logger.info('')
        self.logger.info('Training of neural network:')
        self.logger.info('===========================')
        self.logger.info('')

        if type(self.optimizer) == optim.LBFGS:
            batch_size = max(len(training_data), len(validation_data))

        training_data = CustomDataset(training_data)
        training_sampler = utils.data.RandomSampler(training_data)
        training_loader = utils.data.DataLoader(training_data, batch_size=batch_size,
                                                sampler=training_sampler)

        validation_data = CustomDataset(validation_data)
        validation_sampler = None
        validation_loader = utils.data.DataLoader(validation_data, batch_size=batch_size,
                                                  sampler=validation_sampler)

        # train the neural network
        return self.train_network(training_loader, validation_loader,
                                  number_of_epochs=number_of_epochs, batch_size=batch_size,
                                  learning_rate=learning_rate, show_progress_bar=show_progress_bar)

    def train_network(self, training_loader, validation_loader, number_of_epochs=1000,
                      batch_size=25, learning_rate=None, show_progress_bar=True):
        """Performs actual training of the neural network.

        Parameters
        ----------
        training_loader
            Data loader holding the training data and performing
            the (random) mini-batching of the training data.
        validation_loader
            Data loader holding the validation data and performing
            the (random) mini-batching of the validation data.
        number_of_epochs
            Maximum number of training epochs to perform.
        batch_size
            Batch size to use for mini-batching (has to be the size of the
            training set when using L-BFGS as optimizer).
        learning_rate
            Initial learning rate for the each of the parameter groups of the
            optimizer (here only required for printing of the parameters).
        show_progress_bar
            Determines whether or not to show a progress bar during training.

        Returns
        -------
        Minimum validation and training loss (if early stopping
        scheduler is used).
        """
        number_of_training_samples = len(training_loader.dataset)
        number_of_validation_samples = 0
        if validation_loader:
            number_of_validation_samples = len(validation_loader.dataset)

        # print training and network parameters
        self.print_parameters(number_of_training_samples, batch_size, number_of_epochs,
                              training_loader, learning_rate=learning_rate,
                              number_of_validation_samples=number_of_validation_samples,
                              validation_loader=validation_loader)

        if show_progress_bar:
            bar = ProgressTraining(number_of_epochs, prefix='Train the network:',
                                   suffix='epochs completed')

        phases = ['train']
        dataloaders = {'train':  training_loader}
        phases.append('val')
        dataloaders['val'] = validation_loader

        # perform actual training iteration
        for epoch in range(number_of_epochs):
            losses = {}
            for phase in phases:
                # set state of network according to current phase (training or validation)
                if phase == 'train':
                    self.network.train()
                else:
                    self.network.eval()

                running_loss = 0.0

                # iterate over all batches in the respective phase
                for batch in dataloaders[phase]:
                    # get inputs and targets
                    inputs = batch[0]
                    targets = batch[1]

                    with torch.set_grad_enabled(phase == 'train'):
                        # define closure (especially for optimizers like L-BFGS this is required)
                        def closure():
                            if torch.is_grad_enabled():
                                self.optimizer.zero_grad()
                            # get outputs to current inputs with current network weights and biases
                            outputs = self.network(inputs)
                            # compute loss
                            loss = self.loss_function(outputs, targets)
                            # back propagate loss if necessary
                            if loss.requires_grad:
                                loss.backward()
                            # return loss
                            return loss

                        # perform step of optimizer if in training phase
                        if phase == 'train':
                            self.optimizer.step(closure)

                        # compute current loss
                        loss = closure()

                        # perform step of learning rate scheduler if necessary
                        if self.lr_scheduler:
                            self.lr_scheduler.step()

                    # update current loss
                    running_loss += loss.item() * len(inputs)

                # update loss in current epoch
                epoch_loss = running_loss / len(dataloaders[phase].dataset)

                losses[phase] = epoch_loss

                # perform validation
                if phase == 'val':
                    # check if early stopping is possible
                    if self.es_scheduler and self.es_scheduler(losses['val'], losses['train']):
                        self.logger.info('')
                        self.logger.info('')
                        self.logger.info('Early stopping ...')
                        if hasattr(self.es_scheduler, 'best_loss'):
                            self.logger.info(f'Minimum validation loss: {self.es_scheduler.best_loss}')
                        return self.es_scheduler.best_loss, self.es_scheduler.training_loss

                if np.isnan(losses[phase]):
                    self.logger.info('')
                    self.logger.info('')
                    self.logger.info('Stopping because loss is NAN ...')
                    return self.es_scheduler.best_loss, self.es_scheduler.training_loss

            if show_progress_bar:
                bar.update(losses['train'], losses['val'])

        self.logger.info('')

        if self.es_scheduler:
            return self.es_scheduler.best_loss, self.es_scheduler.training_loss
        else:
            return None

    def print_parameters(self, number_of_training_samples, batch_size, number_of_epochs,
                         training_loader, learning_rate=None, number_of_validation_samples=0,
                         validation_loader=None):
        """Prints the parameters of the neural network and the training.

        Parameters
        ----------
        number_of_training_samples
            Number of training samples used during training the network.
        batch_size
            Batch size to use for mini-batching.
        number_of_epochs
            Maximum number of training epochs to perform.
        training_loader
            Data loader holding the training data and performing
            the (random) mini-batching of the training data.
        learning_rate
            Initial learning rate for the each of the parameter groups of the
            optimizer.
        number_of_validation_samples
            Number of validation samples used during training the network.
        validation_loader
            Data loader holding the validation data and performing
            the (random) mini-batching of the validation data.
        """

        self.network.print_parameters()

        self.logger.info('')

        self.logger.info('=> Training parameters:')
        self.logger.info(f'Training samples: {number_of_training_samples}')
        self.logger.info(f'Epochs: {number_of_epochs}')
        self.logger.info(f'Batch size: {training_loader.batch_size}')
        self.logger.info(f'Training loader: {training_loader.__class__.__name__}')
        self.logger.info(f'Mini-batch sampler: {training_loader.sampler.__class__.__name__}')
        self.logger.info(f'Optimizer: {self.optimizer.__class__.__name__}')
        if learning_rate is not None:
            self.logger.info(f'Initial learning rate: {learning_rate}')
        else:
            self.logger.info(f'Initial learning rate: {self.learning_rate}')
        self.logger.info(f'Learning rate scheduler: {self.lr_scheduler.__class__.__name__}')

        self.logger.info('')

        self.logger.info('=> Validation parameters:')
        self.logger.info('Validation samples: {number_of_validation_samples}')
        self.logger.info(f'Batch size: {validation_loader.batch_size}')
        self.logger.info(f'Validation loader: {validation_loader.__class__.__name__}')
        self.logger.info(f'Mini-batch sampler: {validation_loader.sampler.__class__.__name__}')
        if self.es_scheduler:
            self.logger.info(f'Early stopping scheduler: {self.es_scheduler.__class__.__name__}')
            if hasattr(self.es_scheduler, 'patience'):
                self.logger.info(f'Patience of early stopping scheduler: {self.es_scheduler.patience}')
            if hasattr(self.es_scheduler, 'maximum_loss'):
                self.logger.info(f'Maximum loss to stop with: {self.es_scheduler.maximum_loss}')
        else:
            self.logger.info('No early stopping used')

        self.logger.info('')
