import json
import random

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from actions.prediction.predict import convert_str_to_options
from experiments.eval_data_augmentation import get_prediction

if __name__ == "__main__":
    # ds = "covid_fact"
    ds = "ecqa"
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    model_name = "mistralai/Mistral-7B-v0.1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:0',
                                                 load_in_4bit=True)

    model.config.pad_token_id = model.config.eos_token_id
    num_shot = 3

    if ds == "ecqa":
        df = pd.read_csv("../data/ECQA_dataset.csv")
        questions = list(df["texts"])
        choices = list(df["choices"])
        random_list = random.sample(range(0, len(questions)), 100)

        json_list = []
        for idx in tqdm(random_list):
            pre_prediction = int(get_prediction(tokenizer, model, idx, questions[idx], choices[idx], ds)) - 1

            prompt_template = "You are presented with a multiple-choice question and its options. Generate a " \
                              "counterfactual statement for the given question."
            prompt_template += f"Modify only the question such that {choices[idx].split('-')[pre_prediction]} will " \
                               f"not be selected.\n"
            prompt_template += f"question: {questions[idx]}\n"
            prompt_template += f"choices: {convert_str_to_options(choices[idx])}\n"

            input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.to(model.device.type)
            with torch.no_grad():
                output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40,
                                        max_new_tokens=128)
            result = tokenizer.decode(output[0]).split(prompt_template)[1][:-4]

            post_prediction = int(get_prediction(tokenizer, model, idx, result, choices[idx], ds)) - 1

            print(pre_prediction, post_prediction)

            agreement = 0 if (pre_prediction != post_prediction) else 1

            json_list.append({
                "idx": idx,
                "agreement": agreement,
                "cfe": result
            })

        jsonString = json.dumps(json_list)
        jsonFile = open(f"../cache/{ds}/{ds}_cfe_{model_name.split('/')[1]}.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()
