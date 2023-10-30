from transformers import BertForSequenceClassification, BertTokenizer
from torch import nn
import torch
import numpy as np

# DailyDialog data:
# https://huggingface.co/datasets/daily_dialog
# labels: __dummy__ (0), inform (1), question (2), directive (3), commissive (4)

model_id2label = {0: 'dummy', 1: 'inform', 2: 'question', 3: 'directive', 4: 'commissive'}


class DANetwork(nn.Module):
    def __init__(self):
        super(DANetwork, self).__init__()
        self.bert_emb_size = 768
        self.hidden_dim = 128
        self.output_classes = 5
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.output_classes)

    def forward(self, input_ids, input_mask):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # tensor.to(device) only returns a copy
        # https://stackoverflow.com/questions/54155969/pytorch-instance-tensor-not-moved-to-gpu-even-with-explicit-cuda-call
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        self.bert.to(device)
        output = self.bert(input_ids, attention_mask=input_mask).logits
        return output


class DADataset:
    def __init__(self, samples):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.max_seq_length = 256
        self.input_ids = []
        self.input_masks = []
        self.labels = []

        for sample in samples:
            sample_text, label = sample
            ids, mask = self.get_id_with_mask(sample_text)
            self.input_ids.append(ids)
            self.input_masks.append(mask)
            self.labels.append(torch.tensor(label))

    def get_id_with_mask(self, input_text):
        encoded_dict = self.tokenizer.encode_plus(
            input_text.lower(),
            add_special_tokens=True,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return encoded_dict['input_ids'], encoded_dict['attention_mask']

    def __getitem__(self, idx):
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        return self.input_ids[idx].squeeze(), self.input_masks[idx].squeeze(), label

    def __len__(self):
        return len(self.input_ids)

