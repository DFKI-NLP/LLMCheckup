"""Prediction operation."""
import os
import csv
import random

import pandas as pd

from actions.prediction.predict_grammar import COVID_GRAMMAR
from parsing.guided_decoding.gd_logits_processor import GuidedParser, GuidedDecodingLogitsProcessor


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


def get_demonstrations(_id, num_shot, dataset_name):
    """
    sample demonstrations for few-shot prompting
    :param _id: id of the instance or None (for custom input)
    :param num_shot: number of demonstrations
    :param dataset_name: dataset name
    :return: lists of claims, evidences, labels
    """
    if dataset_name == "covid_fact":
        df = pd.read_csv("./data/COVIDFACT_dataset.csv")
        attr = "claims"
    else:
        # TODO
        pass

    rand_ls = []

    if _id is not None:
        for i in range(num_shot):
            temp = random.randint(0, len(list(df[attr])))
            if temp == _id:
                rand_ls.append(temp - 1)
            else:
                rand_ls.append(temp)
    else:
        for i in range(num_shot):
            rand_ls.append(random.randint(0, len(list(df[attr]))))

    if dataset_name == "covid_fact":
        claims = [list(df["claims"])[i] for i in rand_ls]
        evidences = [list(df["evidences"])[i] for i in rand_ls]
        labels = [list(df["labels"])[i] for i in rand_ls]
    else:
        # TODO
        pass

    return claims, evidences, labels


def get_prediction_by_prompt(prompt_template, conversation):
    """
    Get prediction by given prompt
    :param prompt_template: user input prompt
    :param conversation: conversation object
    :return: single prediction
    """
    tokenizer = conversation.decoder.gpt_tokenizer
    model = conversation.decoder.gpt_model

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids

    if conversation.describe.get_dataset_name() == "covid_fact":
        parser = GuidedParser(COVID_GRAMMAR, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1])
    else:
        # TODO
        pass
    guided_preprocessor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])

    generation = model.greedy_search(input_ids, logits_processor=guided_preprocessor,
                                     pad_token_id=model.config.pad_token_id, eos_token_id=parser.eos_token)

    prediction = tokenizer.decode(generation[0]).split(prompt_template)[1].split(" ")[2].split("<s>")[0]

    return prediction


def prediction_generation(data, conversation, _id, num_shot=3):
    """
    prediction generator
    :param data: filtered data
    :param conversation: conversation object
    :param _id: id of instance or None (for custom input)
    :param num_shot: number of demonstrations
    :return: string for prediction operation
    """
    return_s = ''

    if conversation.describe.get_dataset_name() == "covid_fact":
        claim = None
        evidence = None

        if _id is not None:
            for i, feature_name in enumerate(data.columns):
                if feature_name == "claims":
                    claim = data[feature_name].values[0]
                elif feature_name == "evidences":
                    evidence = data[feature_name].values[0]
        else:
            claim, evidence = conversation.custom_input['first_input'], conversation.custom_input['second_input']

        prompt_template = "Each 3 items in the following list contains the claims, evidence and prediction. Your task " \
                          "is to predict the claims based on evidence as one of the labels: REFUTED, SUPPORTED.\n"

        claims, evidences, labels = get_demonstrations(_id, num_shot, conversation.describe.get_dataset_name())

        for i in range(num_shot):
            prompt_template += f"claim: {claims[i]}\n"
            prompt_template += f"evidence: {evidences[i]}\n"
            prompt_template += f"label: {conversation.class_names[labels[i]]}\n"
            prompt_template += "\n"

        prompt_template += f"claim: {claim}\n"
        prompt_template += f"evidence: {evidence}\n"
        prompt_template += f"label: "
    else:
        # TODO
        pass

    print(prompt_template)

    prediction = get_prediction_by_prompt(prompt_template, conversation)

    if _id is not None:
        filter_string = f"<b>id equal to {_id}</b>"

        return_s += f"The instance with <b>{filter_string}</b>:<br>"
    else:
        return_s += "The custom input prediction: <br>"

    if conversation.describe.get_dataset_name() == "covid_fact":
        return_s += f"<b>Claim:</b> {claim}<br>"
        return_s += f"<b>Evidence:</b> {evidence}<br>"
        return_s += "<b>Prediction:</b> "
    else:
        # TODO
        pass

    return_s += f"<span style=\"background-color: #6CB4EE\">{prediction}</span>."

    return_s += "<br>"
    return return_s, prediction


def predict_operation(conversation, parse_text, i, **kwargs):
    """The prediction operation."""
    if conversation.custom_input is not None and conversation.used is False:
        # if custom input is available
        return_s, _ = prediction_generation(None, conversation, None)
        return return_s, 1

    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    _id = handle_input(parse_text)

    if len(data) == 1:
        # `filter id and predict [E]`
        return_s, _ = prediction_generation(data, conversation, _id)
    else:
        raise ValueError("Too many ids are given!")
    return return_s, 1
