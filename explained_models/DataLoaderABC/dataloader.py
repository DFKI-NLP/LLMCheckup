from abc import ABC, abstractmethod


class DataloaderABC(ABC):
    def __init__(self, tokenizer):
        pass

    @abstractmethod
    def get_dataloader(self):
        pass
