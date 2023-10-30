import json
import os
import random
import time

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# labels = {0: 'false', 1: 'true'}
labels = {0: 'no', 1: 'yes'}
# df = pd.read_csv("../../data/offensive_val.csv")
df = pd.read_csv("/home/qwang/InterroLang/data/offensive_val.csv")
instances = list(df["text"])

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

# few_shot = True
few_shot = False
num_few_shots = [3, 5, 10, 15, 20]
# num_few_shots = [2]

balanced = False


# df = pd.read_csv("../../data/offensive_train.csv")
df = pd.read_csv("/home/qwang/InterroLang/data/offensive_train.csv")
dialogs = list(df["text"])
acts = list(df["label"])


def fill_instance_in_prompt(instance):
    # return f"Instance: {instance}. Given labels: true, false. Based on instance, predict the instance as one of labels. Prediction:"
    return f"Instance: {instance}. Based on instance, predict if the instance is offensive. Answer yes or no:"

if few_shot:
    for num_few_shot in num_few_shots:
        json_list = []
        # for idx, instance in list(enumerate(instances[:10])):
        for idx, instance in list(enumerate(instances)):

            prompt = ""
            temp = random.randint(0, len(dialogs)-1)
            for num in range(num_few_shot):
                rand_num = random.randint(0, len(dialogs) - 1)

                if balanced:
                    counter_0 = 0
                    counter_1 = 0
                    if counter_0 >= num_few_shot // 2:
                        while acts[rand_num] != 1:
                            rand_num = random.randint(0, len(dialogs) - 1)
                    elif counter_1 >= num_few_shot // 2:
                        while acts[rand_num] != 0:
                            rand_num = random.randint(0, len(dialogs) - 1)
                prompt += fill_instance_in_prompt(dialogs[rand_num]) + " " + labels[acts[rand_num]] + "\n"

            prompt += fill_instance_in_prompt(instance)

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            input_ids = input_ids.to(device=device)
            attention_mask = attention_mask.to(device)

            generation = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_p=0.7
            )
            decoded_generation = tokenizer.decode(generation[0], skip_special_tokens=True)
            decoded_generation = decoded_generation.split("\n")[-1]

            temp = decoded_generation[-3:].lower()

            if temp in "no" or "no" in temp:
                prediction = "false"
            elif temp == "yes":
                prediction = "true"
            else:
                prediction = "dummy"

            print({
                "idx": idx,
                "prediction": prediction
            })

            json_list.append({
                "idx": idx,
                "prediction": prediction
            })
        jsonString = json.dumps(json_list)
        jsonFile = open(f"/netscratch/qwang/new/olid_gpt-neo-2.7b_few_shot_{num_few_shot}_prediction.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()
else:
    json_list = []
    for idx, instance in list(enumerate(instances)):
        prompt = fill_instance_in_prompt(instance)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        input_ids = input_ids.to(device=device)
        attention_mask = attention_mask.to(device)

        generation = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.7
        )
        decoded_generation = tokenizer.decode(generation[0], skip_special_tokens=True)

        temp = decoded_generation[-3:].lower()
        if temp in "no" or "no" in temp:
            prediction = "false"
        elif temp == "yes":
            prediction = "true"
        else:
            prediction = "dummy"

        print({
            "idx": idx,
            "prediction": prediction
        })

        json_list.append({
            "idx": idx,
            "prediction": prediction
        })
    jsonString = json.dumps(json_list)
    jsonFile = open(f"/netscratch/qwang/new/olid_gpt-neo-2.7b_zero_shot_prediction.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()