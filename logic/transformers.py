import json
import numpy as np
from torch.nn import Module
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from logic.utils import read_precomputed_explanation_data, read_precomputed_prediction


class TransformerModel(Module):
    def __init__(self, model_id, name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.dataset_name = name

    def __call__(self, *args, **kwargs):
        self.model(*args, **kwargs)

    def predict(self, data, text, conversation=None):
        """ Mirrors the sklearn predict function https://scikit-learn.org/stable/glossary.html#term-predict
        Arguments:
            data: Pandas DataFrame containing columns of text data
            text: preprocessed parse_text
        """
        MAPPING = {"true": 1, "false": 0, "dummy": -1}
        # json_list = read_precomputed_explanation_data(self.dataset_name)
        json_list = read_precomputed_prediction(conversation)
        # Get indices of dataset to filter json_list with
        if data is not None:
            data_indices = data.index.to_list()

        if text is None:
            temp = []
            for item in json_list:
                if item["idx"] in data_indices:
                    temp.append(MAPPING[item["prediction"]])

            return np.array(temp)
        else:
            res = list([MAPPING[json_list[text]["prediction"]]])
            return np.array(res)
