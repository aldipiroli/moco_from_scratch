from abc import ABC


class BaseLoss(ABC):
    def __init__(self, config):
        self.config = config
