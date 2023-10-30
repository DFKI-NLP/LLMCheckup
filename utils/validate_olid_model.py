# -*- coding: utf-8 -*-

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("sinhala-nlp/mbert-olid-en")
tokenizer = AutoTokenizer.from_pretrained("sinhala-nlp/mbert-olid-en")

df = pd.read_csv("../data/offensive_val.csv")

X = np.array(df["text"])
Y = np.array(df["label"])

assert len(X) == len(Y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
predictions = []

for _input in tqdm(X):
    encoding = tokenizer.encode_plus(_input, add_special_tokens=True, return_tensors='pt')
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    input_model = {
        'input_ids': input_ids.long()[None, :],
        'attention_mask': attention_mask.long()[None, :],
    }
    output = torch.argmax(model(**input_model).logits, dim=1)
    predictions.append(output.item())

predictions = np.array(predictions)

print(f"Validation accuracy is: {round(np.sum(predictions == Y) / len(Y) * 100, 2)}%")
