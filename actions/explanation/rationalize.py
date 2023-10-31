import random

from actions.prediction.predict import prediction_generation
import pandas as pd


def get_results(dataset_name, data_path):
    """
    Get the rationlize result

    Args:
        data_path: path to json file
    Returns:
        results: results in csv format
    """
    path = data_path + dataset_name + "/dolly-rationales.csv"
    results = pd.read_csv(path)

    return results


def get_few_shot_str(csv_filename, num_shots=3):
    few_shot_str = ""
    gpt_rationales = pd.read_csv(csv_filename).sample(frac=1).reset_index()
    for i, row in gpt_rationales.iterrows():
        few_shot_str += row["prompt"] + row["completion"] + "\n"
        if i == num_shots - 1:
            break
    return few_shot_str


def formalize_output(dataset_name, text):
    return_s = ""
    if dataset_name == "boolq":
        return_s += "<b>"
        return_s += text[0: 8]
        return_s += "</b>"

        idx_p = text.index("Passage")

        return_s += text[8: idx_p]
        return_s += "<br>"
        return_s += "<b>"
        return_s += text[idx_p: idx_p + 8]
        return_s += "</b>"
        return_s += text[idx_p + 8:]
    elif dataset_name == "daily_dialog":
        return_s += "<b>"
        return_s += text[0: 7]
        return_s += "</b>"
        return_s += text[7:]
    else:
        return_s += "<b>"
        return_s += text[0: 6]
        return_s += "</b>"
        return_s += text[6:]
    return return_s


# @timeout(60)
def rationalize_operation(conversation, parse_text, i, simulation, data_path="./cache/", **kwargs):
    # TODO: Custom input – if conversation.used and conversation.custom_input:

    id_list = []
    for item in parse_text:
        try:
            if type(int(item)) == int:
                id_list.append(int(item))
        except ValueError:
            pass

    model = conversation.decoder.gpt_model
    tokenizer = conversation.decoder.gpt_tokenizer

    df = pd.read_csv("./cache/olid/GPT-4_rationales_OLID_val_132.csv")
    prompts = list(df["prompt"])
    completions = list(df["completion"])

    df = pd.read_csv("./data/offensive_val.csv")
    instances = list(df["text"])
    labels = list(df["label"])

    int2str = {0: "non-offensive", 1: "offensive"}

    step_by_step = True
    num_few_shots = 3

    # additional_prompt = "Let's think step by step."
    additional_prompt = "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry " \
                        "out the plan and solve the problem step by step."

    return_s = ""

    if conversation.custom_input is not None:
        id_list = [1]

    for idx in id_list:
        prompt = ""

        if conversation.custom_input is not None:
            instance = conversation.custom_input
            label = prediction_generation(None, conversation, None)
            if label == "non-offensive":
                label = 0
            else:
                label = 1
        else:
            instance = instances[idx]
            label = labels[idx]

        for i in range(num_few_shots):
            rand_num = random.randint(0, len(completions) - 1)
            if step_by_step:
                prompt += f"{prompts[rand_num]} {additional_prompt} {completions[rand_num]}"
                # prompt += f"{prompts[rand_num]} {additional_prompt} {completions[rand_num]}"
                prompt += "\n"
            else:
                prompt += prompts[rand_num]
                prompt += completions[rand_num]
                prompt += "\n"

            prompt += f"Tweet: {instance}. Based on the tweet, the predicted label is {int2str[label]}. Without " \
                      f"revealing the predicted label in your response, explain why:"
            if step_by_step:
                prompt += additional_prompt
            print("========= Prompt =========")
            print(prompt)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

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

        res = ""
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
                raise Exception("Decoding error!")

        if res != "":
            temp = "<b>Original text:</b> " + instance \
                        + "<br><b>Prediction:</b> " + int2str[label] \
                        + "<br><b>Explanation:</b> " + res
            return_s += temp
        else:
            return_s += "Decoding Error! Please try again!"

        if conversation.custom_input:
            break
    return return_s, 1
