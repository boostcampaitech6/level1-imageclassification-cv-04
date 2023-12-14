from abc import abstractmethod

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, optimizer, config):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config

        self.start_epoch = 1
        self.checkpoint_dir = self.config.model_dir


    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self._train_epoch(epoch)

