import json
import pandas as pd
import matplotlib.pyplot as plt

dataset_name = "covid"

if dataset_name == "covid":

    claims = []
    labels = []
    evidences = []
    links = []

    label2int = {"REFUTED": 1, "SUPPORTED": 0}

    # Open the JSONL file
    with open("./COVIDFACT_dataset.jsonl", 'r') as file:
        for index, line in enumerate(file):
            try:
                data = json.loads(line)

                claims.append(data["claim"])
                labels.append(label2int[data["label"]])
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

    # REFUTED
    counter_0 = 0

    # SUPPORTED
    counter_1 = 1

    for l in labels:
        if l == 0:
            counter_1 += 1
        else:
            counter_0 += 1

    text_labels = ["REFUTED", "SUPPORTED"]
    total = counter_0 + counter_1
    fig, ax = plt.subplots()
    size = [counter_0 / total, counter_1 / total]
    ax.pie(size, labels=text_labels, autopct='%1.1f%%', shadow=True, startangle=90, colors=["lightcoral", "cornflowerblue"])
    plt.show()
    name_dict = {
        "claims": claims,
        "labels": labels,
        "evidences": evidences,
        "links": links,
    }

    df = pd.DataFrame(name_dict)
    df.to_csv('./COVIDFACT_dataset.csv', encoding='utf-8')
else:
    df = pd.read_csv("./cqa_data_train.csv")

    texts = df["q_text"]
    positive_explanations = df["taskA_pos"]
    negative_explanations = df["taskA_neg"]
    free_flow_explanations = df["taskB"]

    choices = []
    answers = []
    idx = [i for i in range(1, 6)]

    for i in range(len(texts)):
        choice = []
        for j in idx:
            choice.append(df["q_op" + str(j)][i])
        ans = df["q_ans"][i]
        answers.append(choice.index(ans))
        choices.append("-".join(choice))

    name_dict = {
        "texts": texts,
        "choices": choices,
        "answers": answers,
        "positive_explanations": positive_explanations,
        "negative_explanations": negative_explanations,
        "free_flow_explanations": free_flow_explanations

    }
    df = pd.DataFrame(name_dict)
    df.to_csv('./ECQA_dataset.csv', encoding='utf-8')
