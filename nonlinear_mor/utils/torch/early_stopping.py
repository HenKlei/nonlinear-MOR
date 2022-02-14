import os


class SimpleEarlyStoppingScheduler:
    """Class that performs checks for early stopping in a simple fashion.

    Parameters
    ----------
    trainer
        Trainer that is used for training.
    checkpoint_filepath
        Path to save the checkpoints to.
    patience
        Number of epochs with non-decreasing validation loss the early stopping
        scheduler is supposed to wait before suggesting the trainer to abort
        the training iteration.
    delta
        Amount of required decrease in the validation loss.
    maximum_loss
        If validation loss is above this value, no early stopping is performed.
    """
    def __init__(self, trainer, checkpoint_filepath=None,
                 patience=10, delta=0., maximum_loss=None):
        super().__init__()
        self.trainer = trainer
        self.checkpoint_filepath = checkpoint_filepath
        if self.checkpoint_filepath:
            os.makedirs(os.path.dirname(self.checkpoint_filepath), exist_ok=True)
        self.patience = patience
        self.delta = delta
        self.maximum_loss = maximum_loss

        self.best_loss = None
        self.training_loss = None
        self.best_network = None
        self.counter = 0

    def __call__(self, validation_loss, training_loss):
        """Check if training should be aborted due to non-decreasing validation loss
           (early stopping)."""
        if self.best_loss is None:
            self.best_loss = validation_loss
            self.training_loss = training_loss
            self.best_network = self.trainer.network
        elif self.best_loss + self.delta <= validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                if self.maximum_loss:
                    if self.maximum_loss > self.best_loss:
                        return True
                else:
                    return True
        else:
            self.best_loss = validation_loss
            self.training_loss = training_loss
            self.best_model = self.trainer.network
            self.counter = 0

        return False
