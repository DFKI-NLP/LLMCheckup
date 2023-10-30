from abc import ABC

from explained_models.ModelABC.model import Model

from transformers import AutoModelForSequenceClassification

DEFAULT_MODEL_ID = "andi611/distilbert-base-uncased-qa-boolq"


class DistilbertQABoolModel(Model, ABC):
    def __init__(self, dataloader, num_labels, model_id=DEFAULT_MODEL_ID):
        super(Model).__init__()
        self.model_id = model_id
        self.dataloader = dataloader.dataloader
        self.tokenizer = dataloader.tokenizer
        self.num_labels = num_labels
        self.model = None

        # Initialize the model and set model's tokenizer
        self.create_model(self.num_labels)
        self.set_tokenizer(self.tokenizer)

    def create_model(self, num_labels):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=num_labels)
        self.set_tokenizer(self.tokenizer)

    def set_tokenizer(self, tokenizer):
        self.model.tokenizer = tokenizer
