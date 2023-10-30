import json
import random

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

df = pd.read_csv("/home/qwang/InterroLang/GPT-4_rationales_OLID_val_132.csv")

prompts = list(df["prompt"])
completions = list(df["completion"])


df = pd.read_csv("/home/qwang/InterroLang/data/offensive_val.csv")
instances = list(df["text"])
labels = list(df["label"])

int2str = {0: "non-offensive", 1: "offensive"}

json_list = []
count_err = 0
for idx in range(300):
    instance = instances[idx]
    label = labels[idx]

    num_few_shots = 3
    step_by_step = False
    few_shot = True

    prompt = ""
    if few_shot:
        for i in range(num_few_shots):
            rand_num = random.randint(0, len(completions) - 1)
            if step_by_step:
                # prompt += f"{prompts[rand_num]} Let's think step by step. {completions[rand_num]}"
                prompt += f"{prompts[rand_num]} Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step. {completions[rand_num]}"
                prompt += "\n"
            else:
                prompt += prompts[rand_num]
                prompt += completions[rand_num]
                prompt += "\n"

        prompt += f"Tweet: {instance}. Based on the tweet, the predicted label is {int2str[label]}. Without revealing the predicted label in your response, explain why: "
        if step_by_step:
            # prompt += "Let's think step by step."
            prompt += "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step."
        print("========= Prompt =========")
        print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    input_ids = input_ids.to(device=device)
    attention_mask = attention_mask.to(device)

    # input_ids = input_ids.to(device=device)
    # attention_mask = attention_mask.to(device)

    generation = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=2048,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=0.95,
        top_p=0.95
    )

    decoded_generation = tokenizer.decode(generation[0], skip_special_tokens=True)

    try:
        res = decoded_generation.split(prompt)[1]
    except:
        try:
            decoded_list = decoded_generation.split("\n")
            counter = -1
            for i in decoded_list[::-1]:
                if i != "":
                    break
                else:
                    counter -= 1
            res = decoded_list[counter]
        except:
            count_err += 1
            res = "error"
            print(count_err)
    print({
        "idx": idx,
        "rationalization": res
    })

    json_list.append({
        "idx": idx,
        "rationalization": res
    })

jsonString = json.dumps(json_list)
jsonFile = open(f"/netscratch/qwang/rationalization_few_shot_gpt-j-6b.json", "w")
jsonFile.write(jsonString)
jsonFile.close()