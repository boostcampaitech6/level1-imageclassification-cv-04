import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, drop_last, pin_memory=True):

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'drop_last': drop_last,
            'pin_memory': pin_memory,
        }
        super().__init__(**self.init_kwargs)