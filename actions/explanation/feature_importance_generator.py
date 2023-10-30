import inseq
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import json

model_name = "EleutherAI/gpt-neo-2.7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = inseq.load_model(model_name, "integrated_gradients")

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")

# df = pd.read_csv("../../data/offensive_val.csv")
df = pd.read_csv("/home/qwang/InterroLang/data/offensive_val.csv")
instances = list(df["text"])
labels = list(df["label"])

json_list = []

int2str = {0: "non-offensive", 1: "offensive"}

for i in range(len(instances)):

    out = model.attribute(
        input_texts=f"{instances[i]} is",
        generated_texts=f"{instances[i]} is {int2str[labels[i]]}",
        n_steps=100,
        internal_batch_size=5
    )
    res = out.aggregate()
    attribution = res.sequence_attributions[0].target_attributions[:, 0].tolist()
    temp_idx = np.where(np.isnan(attribution))[0][0] - 1

    attribution = attribution[:temp_idx]

    tokens = res.sequence_attributions[0].source

    texts = []
    for t in tokens[:-1]:
        texts.append(t.token)

    print(i)
    json_list.append({
        "idx": i,
        "attribution": attribution,
        "text": texts
    })
jsonString = json.dumps(json_list)
jsonFile = open(f"/netscratch/qwang/gpt-neo-2.7B_feature_attribution.json", "w")
jsonFile.write(jsonString)
jsonFile.close()
