import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

model_id = "andi611/distilbert-base-uncased-qa-boolq"

boolq = load_dataset("super_glue", "boolq", split="validation")

predictions = []
corrects = 0
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
for instance in tqdm(boolq, total=len(boolq)):
    encodings = tokenizer(instance['question'], instance['passage'],
                          padding=True,
                          truncation=True,
                          return_tensors='pt')
    out = model(**encodings)
    pred = int(torch.argmax(out.logits, axis=1))
    predictions.append(pred)
    if instance['label'] == pred:
        corrects += 1
print(model_id)
print(corrects/len(predictions))

df = pd.DataFrame(predictions)
model_id_str = model_id.replace('/', '_')
df.to_csv(f"boolq_test_{model_id_str}.csv")
