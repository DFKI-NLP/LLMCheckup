import json
from os import listdir
from os.path import isfile, join

files = [f for f in listdir("../feedback") if isfile(join("../feedback", f))]
json_files = [f for f in files if (f.endswith(".json") and not f.endswith("boolq.json") and not f.endswith("olid.json") and not f.endswith("daily_dialog.json"))]

actions = ["self", "adversarial", "augment", "cfe", "data", "nlpattribute", "filter", "function", "important",
           "keywords", "label", "likelihood", "mistake", "model", "countdata", "predict",
           "randompredict", "rationalize", "score", "show", "similar"]

for file in json_files:

    fileObject = open(f'../feedback/{file}', "r")
    jsonContent = fileObject.read()
    feedback_ls = json.loads(jsonContent)

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


    def process_single_action(f, parsed_text, dataset_ls):
        idx = actions.index(parsed_text)
        feedback_text = f["feedback_text"]
        dataset_ls[idx][feedback_text] += 1


    for f in feedback_ls:
        parsed_text = f["parsed_text"]

        if " " not in parsed_text:
            if f["dataset"] == "boolq":
                process_single_action(f, parsed_text, boolq_ls)
            elif f["dataset"] == "daily_dialog":
                process_single_action(f, parsed_text, dd_ls)
            else:
                process_single_action(f, parsed_text, olid_ls)
        else:
            parsed_text = parsed_text.split(" ")
            for p in parsed_text:
                if p in actions:
                    if f["dataset"] == "boolq":
                        process_single_action(f, p, boolq_ls)
                    elif f["dataset"] == "daily_dialog":
                        process_single_action(f, p, dd_ls)
                    else:
                        process_single_action(f, p, olid_ls)

    boolq_json_res = {}
    olid_json_res = {}
    dd_json_res = {}

    file = file[:file.index(".")]

    for i in range(len(actions)):
        boolq_json_res[actions[i]] = boolq_ls[i]
        olid_json_res[actions[i]] = olid_ls[i]
        dd_json_res[actions[i]] = dd_ls[i]

    jsonString = json.dumps([boolq_json_res])
    jsonFile = open(f"../feedback/{file}-boolq.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    jsonString = json.dumps([olid_json_res])
    jsonFile = open(f"../feedback/{file}-olid.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    jsonString = json.dumps([dd_json_res])
    jsonFile = open(f"../feedback/{file}-daily_dialog.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()
