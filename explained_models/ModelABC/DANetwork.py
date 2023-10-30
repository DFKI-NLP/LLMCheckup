import json
import numpy as np
import torch
from torch import nn
from transformers import BertForSequenceClassification

DEFAULT_MODEL_ID = "bert-base-uncased"


class DANetwork(nn.Module):
    def __init__(self, bert_emb_size=768, hidden_dim=128, model_id=DEFAULT_MODEL_ID, num_labels=5):
        super(DANetwork, self).__init__()
        self.bert_emb_size = bert_emb_size
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.bert = None
        self.model_id = model_id
        self.create_model()
        if not torch.cuda.is_available():
            self.load_state_dict(torch.load('./explained_models/da_classifier/saved_model/5e_5e-06lr',
                                            map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load('./explained_models/da_classifier/saved_model/5e_5e-06lr'))
        self.dataset_name = "daily_dialog"

    def forward(self, input_ids, input_mask):
        output = self.bert(input_ids, attention_mask=input_mask).logits
        return output

    def create_model(self):
        self.bert = BertForSequenceClassification.from_pretrained(self.model_id, num_labels=self.num_labels)

    def predict(self, data, text):
        path = f"./cache/{self.dataset_name}/ig_explainer_{self.dataset_name}_explanation.json"
        try:
            fileObject = open(path, "r")
            jsonContent = fileObject.read()
            json_list = json.loads(jsonContent)
        except:
            raise FileNotFoundError(f"The required cache with path {path} doesn't exist!")

        if text is None:
            temp = []
            for item in json_list:
                temp.append(item["label"])

            return np.array(temp)
        else:
            res = list([json_list[text]["label"]])
            return np.array(res)
