import torch.utils as utils


class CustomDataset(utils.data.Dataset):
    """Class that represents the dataset to use in PyTorch.

    Parameters
    ----------
    training_data
        Set of training parameters and the respective coefficients of the
        solution in the reduced basis.
    """

    def __init__(self, training_data):
        self.training_data = training_data

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        t = self.training_data[idx]
        return t
