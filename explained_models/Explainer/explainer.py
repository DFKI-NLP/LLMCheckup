from abc import ABC, abstractmethod


class Explainer(ABC):

    @abstractmethod
    def generate_explanation(self):
        pass
    