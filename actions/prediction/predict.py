"""Prediction operation."""
import random

import pandas as pd

import requests

import torch

from actions.prediction.predict_grammar import COVID_GRAMMAR, ECQA_GRAMMAR
from parsing.guided_decoding.gd_logits_processor import GuidedParser, GuidedDecodingLogitsProcessor

from googlesearch import search


def handle_input(parse_text):
    num = None
    for item in parse_text:
        try:
            if int(item):
                num = int(item)
        except:
            pass
    return num


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
        df = pd.read_csv("./data/ECQA_dataset.csv")
        attr = "texts"

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
        return claims, evidences, labels
    else:
        questions = [list(df["texts"])[i] for i in rand_ls]
        choices = [list(df["choices"])[i] for i in rand_ls]
        labels = [list(df["answers"])[i] for i in rand_ls]
        return questions, choices, labels


def get_prediction_by_prompt(prompt_template, conversation, choice=None):
    """
    Get prediction by given prompt
    :param choice:
    :param prompt_template: user input prompt
    :param conversation: conversation object
    :return: single prediction
    """
    tokenizer = conversation.decoder.gpt_tokenizer
    model = conversation.decoder.gpt_model

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.to(model.device.type)

    if conversation.describe.get_dataset_name() == "covid_fact":
        parser = GuidedParser(COVID_GRAMMAR, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1])
    else:
        parser = GuidedParser(ECQA_GRAMMAR, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1])
    guided_preprocessor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])

    with torch.no_grad():
        generation = model.greedy_search(input_ids, logits_processor=guided_preprocessor,
                                         pad_token_id=model.config.pad_token_id,
                                         eos_token_id=parser.eos_token, device=model.device.type)

    decoder_name = conversation.decoder.parser_name
    if "falcon" in decoder_name or "pythia" in decoder_name:
        prediction = tokenizer.decode(generation[0]).split(prompt_template)[1].split(" [e]")[0].split(" ")[1]
    else:
        prediction = tokenizer.decode(generation[0]).split(prompt_template)[1].split(" ")[2].split("<s>")[0]

    return prediction


def convert_str_to_options(choice):
    """
    Convert compressed choices to string
    :param choice: compressed choices by '-'
    :return: choices in order
    """
    res = ""
    options = choice.split("-")

    for idx, op in enumerate(options):
        res += f"({idx + 1}) {op} "
    return res


def get_fields_and_prompt(data, conversation, _id, num_shot, given_first_field=None, given_second_field=None):
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
            prompt_template += f"prediction: {conversation.class_names[labels[i]]}\n"
            prompt_template += "\n"

        if given_first_field is not None:
            claim = given_first_field

        if given_second_field is not None:
            evidence = given_second_field

        first_field = claim
        second_field = evidence

        prompt_template += f"claim: {claim}\n"
        prompt_template += f"evidence: {evidence}\n"
        prompt_template += f"prediction: "
    else:
        question = None
        choice = None

        if _id is not None:
            for i, feature_name in enumerate(data.columns):
                if feature_name == "texts":
                    question = data[feature_name].values[0]
                elif feature_name == "choices":
                    choice = data[feature_name].values[0]
        else:
            question, choice = conversation.custom_input['first_input'], conversation.custom_input['second_input']

        prompt_template = "Each 3 items in the following list contains the question, choice and prediction. Your task " \
                          "is to choose one of the choices as the answer for the question\n"

        questions, choices, labels = get_demonstrations(_id, num_shot, conversation.describe.get_dataset_name())

        for i in range(num_shot):
            prompt_template += f"question: {questions[i]}\n"
            prompt_template += f"choices: {convert_str_to_options(choices[i])}\n"
            prompt_template += f"prediction: {labels[i] + 1}\n"
            prompt_template += "\n"

        if given_first_field is not None:
            question = given_first_field

        if given_second_field is not None:
            choice = given_second_field

        first_field = question
        second_field = choice

        prompt_template += f"question: {first_field}\n"
        prompt_template += f"choices: {convert_str_to_options(second_field)}\n"
        prompt_template += f"prediction: "

    return first_field, second_field, prompt_template


def store_prediction(conversation, _id, prediction):
    """
    Store prediction into dataframe
    :param conversation: conversation object
    :param _id: id of instance
    :param prediction: prediction of the given instance
    """
    df = conversation.precomputation_of_prediction

    if conversation.describe.get_dataset_name() == "covid_fact":
        # Reverse the class name dictionary to get prediction in digits
        prediction = {v: k for k, v in conversation.class_names.items()}[prediction]

    # Check if available
    if not any(df["id"] == _id):
        df.loc[len(df)] = {"id": _id, "prediction": int(prediction)}


def prediction_generation(data, conversation, _id, num_shot=3, given_first_field=None, given_second_field=None,
                          external_call=True, external_search=True):
    """
    prediction generator
    :param external_search: external information retrieval
    :param external_call: external call
    :param given_first_field: perturbed text
    :param given_second_field: perturbed text
    :param data: filtered data
    :param conversation: conversation object
    :param _id: id of instance or None (for custom input)
    :param num_shot: number of demonstrations
    :return: string for prediction operation
    """
    return_s = ""

    first_field, second_field, prompt_template = get_fields_and_prompt(data, conversation, _id, num_shot,
                                                                       given_first_field, given_second_field)

    if not external_call:
        conversation.current_prompt = prompt_template

    print(prompt_template)

    prediction = get_prediction_by_prompt(prompt_template, conversation, choice=second_field)

    # Store prediction in precomputation dataframe
    store_prediction(conversation, _id, prediction)

    if _id is not None:
        filter_string = f"<b>id equal to {_id}</b>"

        return_s += f"The instance with <b>{filter_string}</b>:<br>"
    else:
        return_s += "The custom input prediction: <br>"

    if conversation.describe.get_dataset_name() == "covid_fact":
        if given_second_field is not None:
            return_s += f"<b>Perturbed Evidence:</b> {second_field}<br>"
        else:
            return_s += f"<b>Claim:</b> {first_field}<br>"
            return_s += f"<b>Evidence:</b> {second_field}<br>"
        return_s += "<b>Prediction:</b> "
        return_s += f"<span style=\"background-color: #6CB4EE\">{prediction}</span>.<br><br>"

        if external_search:
            try:
                # Do information retrieval
                link_ls = list(search(first_field))
                return_s += f"<b>Potential relevant link</b>: <a href='{link_ls[0]}'>{link_ls[0]}</a>"
            except requests.exceptions.HTTPError:
                pass

        return return_s, prediction
    else:
        # prediction in format: i
        if given_second_field is not None:
            return_s += f"<b>Perturbed Evidence:</b> {second_field}<br>"
        else:
            return_s += f"<b>Question:</b> {first_field}<br>"
            return_s += f"<b>Choices:</b> {convert_str_to_options(second_field)}<br>"
        return_s += "<b>Prediction:</b> "

        return_s += f"<span style=\"background-color: #6CB4EE\">({prediction}) {second_field.split('-')[int(prediction) - 1]}</span>.<br><br>"

        if external_search:
            try:
                # Do information retrieval
                link_ls = list(search(first_field))
                return_s += f"<b>Potential relevant link</b>: <a href='{link_ls[0]}'>{link_ls[0]}</a>"
            except requests.exceptions.HTTPError:
                pass

        # Return index of choice
        return return_s, int(prediction) - 1


def predict_operation(conversation, parse_text, i, **kwargs):
    """The prediction operation."""
    if conversation.custom_input is not None and conversation.used is False:
        # if custom input is available
        return_s, _ = prediction_generation(None, conversation, None, external_call=False)

        return return_s, 1

    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    _id = handle_input(parse_text)

    if len(data) == 1:
        # `filter id and predict [E]`
        return_s, _ = prediction_generation(data, conversation, _id, external_call=False)
    else:
        raise ValueError("Too less/much ids are given!")
    return return_s, 1
