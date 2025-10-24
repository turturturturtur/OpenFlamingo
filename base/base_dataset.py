import torch
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build():
        pass


    def __getitem__(self, index: int):
        pass
    
    def __len__(self):
        return self.labels.shape[0]

    @property
    def num_class(self):
        return self.num_class