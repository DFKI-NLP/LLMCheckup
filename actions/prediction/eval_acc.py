import json

import numpy as np
import pandas as pd
import sklearn.metrics
# labels = {'dummy': 0, 'inform': 1, 'question': 2, 'directive': 3, 'commissive': 4}
labels = {'non-offensive': 0, 'offensive': 1}

# 42.56
# 41.84
# 2.57
# fileObject = open('../../cache/olid_gpt-neo-2.7b_few_shot_3_prediction.json', "r")
#
# # 49.66
# # 46.56
# # 5.51
# fileObject = open('../../cache/olid_gpt-neo-2.7b_few_shot_5_prediction.json', "r")
#
# # 52.83
# # 53.89
# # 8.46
# fileObject = open('../../cache/olid_gpt-neo-2.7b_few_shot_10_prediction.json', "r")
#
# # 55.40
# # 54.61
# # 9.71
# fileObject = open('../../cache/olid_gpt-neo-2.7b_few_shot_15_prediction.json', "r")
#
# # 56.64
# # 56.00
# # 10.54
# fileObject = open('../../cache/olid_gpt-neo-2.7b_few_shot_20_prediction.json', "r")
#
# ###############
# # 5.82
# fileObject = open('../../cache/old/olid_gpt-neo-2.7b_zero_shot_prediction.json', "r")
# # 2.23
# fileObject = open('../../cache/old/olid_gpt-neo-2.7b_few_shot_3_prediction.json', "r")
# # 1.93
# fileObject = open('../../cache/old/olid_gpt-neo-2.7b_few_shot_5_prediction.json', "r")
# # 14.27
# fileObject = open('../../cache/old/olid_gpt-neo-2.7b_few_shot_10_prediction.json', "r")
# # 48.04
# fileObject = open('../../cache/old/olid_gpt-neo-2.7b_few_shot_15_prediction.json', "r")
# # 74.85
# fileObject = open('../../cache/old/olid_gpt-neo-2.7b_few_shot_20_prediction.json', "r")
#
# # 29.42, 18.20
# fileObject = open('../../cache/new/olid_gpt-neo-2.7b_zero_shot_prediction.json', "r")
# # 45.85, 3.36
# fileObject = open('../../cache/new/olid_gpt-neo-2.7b_few_shot_3_prediction.json', "r")
# # 51.36, 6.80 good
# fileObject = open('../../cache/new/olid_gpt-neo-2.7b_few_shot_5_prediction.json', "r")
#
# # 56.57, 7.97
# fileObject = open('../../cache/new/olid_gpt-neo-2.7b_few_shot_10_prediction.json', "r")
# # 60.08 6.42
# fileObject = open('../../cache/new/olid_gpt-neo-2.7b_few_shot_15_prediction.json', "r")
# # 61.97 5.25
# fileObject = open('../../cache/new/olid_gpt-neo-2.7b_few_shot_20_prediction.json', "r")
# 8.57 73.90
# fileObject = open('../../cache/new/olid_gpt-j-6b_zero_shot_prediction.json', "r")
# # 56.12 0.91
# fileObject = open('../../cache/new/olid_gpt-j-6b_few_shot_3_prediction.json', "r")
# # 60.20 1.17
# fileObject = open('../../cache/new/olid_gpt-j-6b_few_shot_5_prediction.json', "r")
# fileObject = open('../../cache/new/olid_gpt-j-6b_few_shot_10_prediction.json', "r")
##################

# 27.76
# 19.98
# fileObject = open('../../cache/olid_gpt-neo-2.7b_zero_shot_prediction.json', "r")

# 21.94
# fileObject = open('../../cache/olid_gpt-j-6b_zero_shot_prediction.json', "r")
#
# # 47.09
# fileObject = open('../../cache/olid_gpt-j-6b_few_shot_3_prediction.json', "r")
#
# # 57.18
# fileObject = open('../../cache/olid_gpt-j-6b_few_shot_5_prediction.json', "r")
#
# # 62.42
# fileObject = open('../../cache/olid_gpt-j-6b_few_shot_10_prediction.json', "r")
#
# # 62.35
# fileObject = open('../../cache/olid_gpt-j-6b_few_shot_15_prediction.json', "r")
#
# # 60.54
# fileObject = open('../../cache/olid_gpt-j-6b_few_shot_20_prediction.json', "r")

# fileObject = open('../../cache/balanced/olid_gpt-neo-2.7b_balanced_few_shot_3_prediction.json', "r")
# fileObject = open('../../cache/balanced/olid_gpt-neo-2.7b_balanced_few_shot_5_prediction.json', "r")
# fileObject = open('../../cache/balanced/olid_gpt-neo-2.7b_balanced_few_shot_10_prediction.json', "r")
# fileObject = open('../../cache/balanced/olid_gpt-neo-2.7b_balanced_few_shot_15_prediction.json', "r")
# fileObject = open('../../cache/balanced/olid_gpt-neo-2.7b_balanced_few_shot_20_prediction.json', "r")

# 65.33 1.92 0: 98.07
fileObject = open('../../cache/guided/olid_gpt-neo-2.7b_zero_shot_prediction.json', "r")
# 60.31 1: 20.85 79.15
fileObject = open('../../cache/guided/olid_gpt-neo-2.7b_3_shot_prediction.json', "r")
# 61.14 16.62 83.38
fileObject = open('../../cache/guided/olid_gpt-neo-2.7b_5_shot_prediction.json', "r")
# 64.27 7.74 92.26
fileObject = open('../../cache/guided/olid_gpt-neo-2.7b_10_shot_prediction.json', "r")
# # 64.80 3.51 96.49
fileObject = open('../../cache/guided/olid_gpt-neo-2.7b_15_shot_prediction.json', "r")
# 65.52 2.34 97.66
fileObject = open('../../cache/guided/olid_gpt-neo-2.7b_20_shot_prediction.json', "r")

# 65.18 1: 1.25 0: 98.75
# fileObject = open('../../cache/guided/olid_gpt-j-6b_zero_shot_prediction.json', "r")

# 60.39 23.04 76.96
fileObject = open('../../cache/guided/olid_gpt-j-6b_3_shot_prediction.json', "r")

# 59.74 23.34 74.66
fileObject = open('../../cache/guided/olid_gpt-j-6b_5_shot_prediction.json', "r")
# 63.33 18.13 81.87
fileObject = open('../../cache/guided/olid_gpt-j-6b_10_shot_prediction.json', "r")
# 64.31 14.20 85.80
fileObject = open('../../cache/guided/olid_gpt-j-6b_15_shot_prediction.json', "r")
# 65.14 13.97 86.03
fileObject = open('../../cache/guided/olid_gpt-j-6b_20_shot_prediction.json', "r")

jsonContent = fileObject.read()
json_ls = json.loads(jsonContent)

predictions = []
ones = 0
zeros = 0

for i in json_ls:
    res = labels[i["prediction"]]
    predictions.append(res)
    if res == 0:
        zeros += 1
    elif res == 1:
        ones += 1

predictions = np.array(predictions)

# df = pd.read_csv("../../data/da_test_set_with_indices.csv")
df = pd.read_csv("../../data/offensive_val.csv")
labels = np.array(list(df["label"]))
print(len(labels), len(predictions))
assert len(labels) == len(predictions)
# 15.84%
print(np.sum(predictions == labels)/len(predictions))

print(f"Ones: {ones/len(predictions)}")
print(f"Zeros: {zeros/len(predictions)}")

# print(sklearn.metrics.f1_score(predictions, labels, average="micro"))
