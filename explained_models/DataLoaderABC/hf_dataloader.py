from abc import ABC

import torch

from explained_models.DataLoaderABC.dataloader import DataloaderABC
from explained_models.boolq.data import get_dataset


class HFDataloader(DataloaderABC, ABC):
    def __init__(self, tokenizer, batch_size, number_of_instance=None):
        super(HFDataloader).__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.dataset = get_dataset(self.tokenizer, number_of_instance)
        self.dataloader = self.get_dataloader()

    def get_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size)
