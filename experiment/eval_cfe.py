import json

import evaluate
import numpy as np
import pandas as pd
from tqdm import trange

df = pd.read_csv("../data/offensive_val.csv")
texts = list(df["text"])

# fileObject = open("../../cache/counterfactual_gpt-neo-2.7b.json", "r")
fileObject = open("../cache/counterfactual_gpt-j-6b.json", "r")
jsonContent = fileObject.read()
json_list = json.loads(jsonContent)

results = []
precisions = []
bleu = evaluate.load("bleu")

# for i in trange(len(texts)):
# # for i in trange(100):
#     if json_list[i]["counterfactual"] != "error":
#         result = bleu.compute(predictions=[json_list[i]["counterfactual"]], references=[[texts[i]]])
#         # results.append(cos_sim)
#         results.append(result["bleu"])
#         precisions.append(result["precisions"][0])
# results = np.array(results)
# precisions = np.array(precisions)
# np.save("../../cache/gpt-neo-2.7-counterfactual-results.npy", results)
# np.save("../../cache/gpt-neo-2.7-counterfactual-precisions.npy", precisions)
# np.save("../../cache/gpt-j-6b-counterfactual-results.npy", results)


# res = np.load("../../cache/gpt-neo-2.7-data-augmentation-results.npy")
res = np.load("../cache/gpt-j-6b-data-augmentation-results.npy")

# 46.91, 47,47
print(np.average(res))

# 95.39 93.68
print(np.max(res))

# -0.04, -0.05
print(np.min(res))