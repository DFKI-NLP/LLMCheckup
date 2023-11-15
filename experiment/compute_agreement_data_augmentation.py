import json
import os

import numpy as np

path_to_dir = "../cache/ecqa"
json_files = os.listdir(path_to_dir)

for i in json_files:
    fileObject = open(f"{path_to_dir}/{i}", "r")
    jsonContent = fileObject.read()
    json_list = json.loads(jsonContent)

    agreements = 0
    claim_sim = []
    evidence_sim = []

    for item in json_list:
        agreements += item["agreement"]
        claim_sim.append(item["claim_cos_sim"])
        # evidence_sim.append(item["evidence_cos_sim"])

    print({
        "File": i,
        "Agreement": agreements/len(json_list),
        "Claim similarity": round(np.average(np.array(claim_sim)), 2),
        # "Evidence similarity": round(np.average(np.array(evidence_sim)), 2)
    })