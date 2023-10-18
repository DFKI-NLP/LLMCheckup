import json
import pandas as pd

claims = []
labels = []
evidences = []
links = []

# Open the JSONL file
with open("./COVIDFACT_dataset.jsonl", 'r') as file:
    for index, line in enumerate(file):
        try:
            data = json.loads(line)

            claims.append(data["claim"])
            labels.append(data["label"])
            links.append(data["gold_source"])
            evidence = data["evidence"]

            temp = ""
            for e in evidence[:-1]:
                temp += e
                temp += " "
            temp += evidence[-1]
            evidences.append(temp)
        except:
            print(line, index)


name_dict = {
    "claims": claims,
    "labels": labels,
    "evidences": evidences,
    "links": links,
  }

df = pd.DataFrame(name_dict)
df.to_csv('./COVIDFACT_dataset.csv', encoding='utf-8')
