from base.base_data_loader import BaseDataLoader


class MaskDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True):
        # super().__init__(dataset, batch_size, shuffle, num_workers, drop_last, pin_memory)
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
