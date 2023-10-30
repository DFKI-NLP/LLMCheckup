import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import trange

df = pd.read_csv("../data/offensive_val.csv")
texts = list(df["text"])

fileObject = open("../cache/data_augmentation_gpt-neo-2.7b.json", "r")
# fileObject = open("../../cache/data_augmentation_gpt-j-6b.json.json", "r")
jsonContent = fileObject.read()
json_list = json.loads(jsonContent)

similarity_model = SentenceTransformer("all-mpnet-base-v2")

results = []

for i in trange(len(texts)):
    if json_list[i]["rephrased text"] != "error":
        query_embedding = similarity_model.encode(texts[i])
        sent_embeddings = similarity_model.encode(json_list[i]["rephrased text"])
        cos_sim = util.cos_sim(query_embedding, sent_embeddings)[0].tolist()[0]
        results.append(cos_sim)

results = np.array(results)
np.save("../cache/gpt-neo-2.7-data-augmentation-results.npy", results)
# np.save("../../cache/gpt-j-6b-data-augmentation-results.npy", results)

# 46.91, 47.47
print(np.average(results))

# res = np.load("../../cache/gpt-neo-2.7-data-augmentation-results.npy")
# res = np.load("../../cache/gpt-j-6b-data-augmentation-results.npy")
#
# # 95.39, 93.68
# print(np.max(res))
#
# # -0.04, -0.05
# print(np.min(res))