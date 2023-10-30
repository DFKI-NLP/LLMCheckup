from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, dataloader, tokenizer):
        pass

    @abstractmethod
    def create_model(self):
        pass
