import json
from os import listdir
from os.path import isfile, join


files = [f for f in listdir("../feedback") if isfile(join("../feedback", f))]
json_files = [f for f in files if (f.endswith(".json") and (f.endswith("boolq.json") or f.endswith("olid.json") or f.endswith("daily_dialog.json")))]

actions = ["self", "adversarial", "augment", "cfe", "data", "nlpattribute", "filter", "function", "important",
           "keywords", "label", "likelihood", "mistake", "model", "countdata", "predict",
           "randompredict", "rationalize", "score", "show", "similar"]

boolq_ls = [
        {"Correctness Positive": 0, "Correctness Negative": 0, "Satisfaction Positive": 0, "Satisfaction Negative": 0,
         "Helpfulness Positive": 0, "Helpfulness Negative": 0} for i in actions]

dd_ls = [
    {"Correctness Positive": 0, "Correctness Negative": 0, "Satisfaction Positive": 0, "Satisfaction Negative": 0,
     "Helpfulness Positive": 0, "Helpfulness Negative": 0} for i in actions]

olid_ls = [
    {"Correctness Positive": 0, "Correctness Negative": 0, "Satisfaction Positive": 0, "Satisfaction Negative": 0,
     "Helpfulness Positive": 0, "Helpfulness Negative": 0} for i in actions]

keys = ["Correctness Positive", "Correctness Negative", "Satisfaction Positive", "Satisfaction Negative",
            "Helpfulness Positive", "Helpfulness Negative"]

for file in json_files:
    if file.endswith("boolq.json"):
        ls = boolq_ls
    elif file.endswith("olid.json"):
        ls = olid_ls
    else:
        ls = dd_ls

    fileObject = open(f'../feedback/{file}', "r")
    jsonContent = fileObject.read()
    feedback_ls = json.loads(jsonContent)[0]

    for key in feedback_ls.keys():
        idx = actions.index(key)
        # print(key)
        for k in keys:
            ls[idx][k] += feedback_ls[key][k]

boolq_dict = {}
olid_dict = {}
da_dict = {}
for i in range(len(boolq_ls)):
    boolq_dict[actions[i]] = boolq_ls[i]
    olid_dict[actions[i]] = olid_ls[i]
    da_dict[actions[i]] = dd_ls[i]

print("boolq", boolq_dict)
print("###########################################")
print(olid_dict)
print("###########################################")
print(da_dict)