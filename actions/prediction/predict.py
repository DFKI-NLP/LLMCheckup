"""Prediction operation."""
import json
import os
import csv
import random

import numpy as np
import pandas as pd
import torch.cuda

from actions.util_functions import gen_parse_op_text, get_parse_filter_text
from parsing.guided_decoding.gd_logits_processor import GuidedParser, GuidedDecodingLogitsProcessor

from timeout import timeout


def handle_input(parse_text):
    num = None
    for item in parse_text:
        try:
            if int(item):
                num = int(item)
        except:
            pass
    return num


def store_results(inputs, predictions, cache_path):
    """
    Store custom inputs and its predictions in csv file
    Args:
        inputs: custom input
        predictions: corresponding predictions
        cache_path: path to cache/csv
    """
    if not os.path.exists(cache_path):
        with open(cache_path, 'w', newline='') as file:
            writer = csv.writer(file)

            # Write header
            writer.writerow(["idx", "Input text", "Prediction"])
            for i in range(len(inputs)):
                writer.writerow([i, inputs[i], predictions[i]])
            file.close()

    else:
        rows = []
        with open(cache_path, 'r', ) as file:
            fieldnames = ["idx", "Input text", "Prediction"]
            reader = csv.DictReader(file, fieldnames=fieldnames)

            for row in reader:
                rows.append(row)
            file.close()
        length = len(rows)

        with open(cache_path, 'w', newline='') as file:
            fieldnames = ["idx", "Input text", "Prediction"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(1, length):
                writer.writerow(rows[i])

            for i in range(len(inputs)):
                writer.writerow({
                    "idx": i + length - 1,
                    "Input text": inputs[i],
                    "Prediction": predictions[i]
                })
            file.close()


def prediction_with_custom_input(conversation):
    """
    Predict the custom input from user that is not contained in the dataset
    Args:
        conversation: Conversation object

    Returns:
        format string with inputs and predictions
    """

    inputs = [conversation.custom_input]

    if len(inputs) == 0:
        return None

    GRAMMAR = r"""
    ?start: action
    action: operation done 

    done: " [e]"

    operation: off | inoff 

    inoff: " non-offensive"

    off: " offensive"
    """
    df = pd.read_csv("./data/offensive_val.csv")
    instances = list(df["text"])

    df = pd.read_csv("./data/offensive_train.csv")
    texts = list(df["text"])
    labels = list(df["label"])

    id2str = {0: "non-offensive", 1: "offensive"}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = conversation.decoder.gpt_tokenizer
    model = conversation.decoder.gpt_model

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    predictions = []
    dataset_name = conversation.describe.get_dataset_name()

    model_name = conversation.decoder.parser_name
    # if model_name == 'EleutherAI/gpt-neo-2.7B':
    #     num_few_shot = 5
    # elif model_name == "EleutherAI/gpt-j-6b":
    #     num_few_shot = 15
    # else:
    #     raise NotImplementedError(f"Model {model_name} is unknown!")

    for string in inputs:
        prompt = ""
        # for num in range(num_few_shot):
        #     rand_num = random.randint(0, len(instances) - 1)
        #
        #     counter_0 = 0
        #     counter_1 = 0
        #     if counter_0 >= num_few_shot // 2:
        #         while labels[rand_num] != 1:
        #             rand_num = random.randint(0, len(instances) - 1)
        #     elif counter_1 >= num_few_shot // 2:
        #         while labels[rand_num] != 0:
        #             rand_num = random.randint(0, len(instances) - 1)
        #
        #     prompt += f"Instance: {texts[rand_num]}\n"
        #     prompt += f"Parsed: {id2str[labels[rand_num]]} [e]\n"
        prompt += f"Instance: {string}\n"
        prompt += "Parsed:"
        torch.cuda.empty_cache()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)

        parser = GuidedParser(GRAMMAR, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1])
        guided_preprocessor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])

        generation = model.greedy_search(input_ids, logits_processor=guided_preprocessor,
                                         pad_token_id=model.config.pad_token_id, eos_token_id=parser.eos_token)

        decoded_generation = tokenizer.decode(generation[0])
        try:
            prediction = decoded_generation.split(prompt)[1].split("[e]")[0].strip()
        except:
            print(decoded_generation)
            temp = decoded_generation[-17:].split("[e]")[0].strip()
            if "non-offensive" in temp:
                prediction = "non-offensive"
            else:
                prediction = "offensive"

        predictions.append(prediction)

    return predictions


def prediction_with_id(model, data, conversation, text):
    """Get the prediction of an instance with ID"""
    return_s = ''

    model_predictions = model.predict(data, text, conversation)

    filter_string = gen_parse_op_text(conversation)

    return_s += f"The instance with <b>{filter_string}</b> is predicted "
    if conversation.class_names is None:
        prediction_class = str(model_predictions[0])
        return_s += f"<b>{prediction_class}</b>"
    else:
        class_text = conversation.class_names[model_predictions[0]]
        return_s += f"<span style=\"background-color: #6CB4EE\">{class_text}</span>."

    return_s += "<br>"
    return return_s


def prediction_on_dataset(model, data, conversation, text):
    """Get the predictions on multiple instances (entire dataset or subset of length > 1)"""
    return_s = ''
    model_predictions = model.predict(data, text, conversation)

    intro_text = get_parse_filter_text(conversation)
    return_s += f"{intro_text} the model predicts:"
    unique_preds = np.unique(model_predictions)
    return_s += "<ul>"
    for j, uniq_p in enumerate(unique_preds):
        return_s += "<li>"
        freq = np.sum(uniq_p == model_predictions) / len(model_predictions)
        round_freq = str(round(freq * 100, conversation.rounding_precision))

        if conversation.class_names is None:
            return_s += f"<b>class {uniq_p}</b>, {round_freq}%"
        else:
            try:
                class_text = conversation.class_names[uniq_p]
            except KeyError:
                class_text = uniq_p
            return_s += f"<span style=\"background-color: #6CB4EE\">{class_text}</span>, {round_freq}%"
        return_s += "</li>"
    return_s += "</ul>"

    return_s += "<br>"
    return return_s


@timeout(60)
def predict_operation(conversation, parse_text, i, **kwargs):
    """The prediction operation."""
    if conversation.custom_input is not None and conversation.used is False:
        predictions = prediction_with_custom_input(conversation)

        if predictions is not None:
            return_s = f"Your input is: <b>{conversation.custom_input}</b> <br>"
            return_s += f"The prediction is <b>{predictions[0]}</b>"
            return return_s, 1

    model = conversation.get_var('model').contents
    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    text = handle_input(parse_text)

    if len(data) == 1:
        # `filter id and predict [E]`
        return_s = prediction_with_id(model, data, conversation, text)
    else:
        # `predict [E]`
        return_s = prediction_on_dataset(model, data, conversation, text)
        #_, return_s = get_prediction_on_temp_dataset(conversation)

    return return_s, 1
