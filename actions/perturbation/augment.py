"""Data augmentation operation."""

import nlpaug.augmenter.word as naw
import torch

from actions.prediction.predict import prediction_generation, convert_str_to_options


def get_augmentation(conversation, original_text):
    """
    Get augmented text based on original text
    :param conversation: conversation object
    :param original_text: original text
    :return: augmented text
    """
    model = conversation.decoder.gpt_model
    tokenizer = conversation.decoder.gpt_tokenizer

    prompt_template = f"Based on '{original_text}', generate a semantic similar one."

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.to(model.device.type)
    with torch.no_grad():
        output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40,
                                max_new_tokens=128)
    result = tokenizer.decode(output[0]).split(prompt_template)[1][:-4]

    return result


def augment_operation(conversation, parse_text, i, **kwargs):
    """Data augmentation."""
    data = conversation.temp_dataset.contents['X']
    idx = None

    if conversation.custom_input is not None and conversation.used is False:
        if conversation.describe.get_dataset_name() == "covid_fact":
            claim, evidence = conversation.custom_input['first_input'], conversation.custom_input['second_input']
        else:
            question, choices = conversation.custom_input['first_input'], conversation.custom_input['second_input']
    else:
        assert len(conversation.temp_dataset.contents["X"]) == 1

        try:
            idx = conversation.temp_dataset.contents["X"].index[0]
        except ValueError:
            return "Sorry, invalid id", 1

        if conversation.describe.get_dataset_name() == "covid_fact":
            claim = conversation.get_var("dataset").contents["X"].iloc[idx]["claims"]
            evidence = conversation.get_var("dataset").contents["X"].iloc[idx]["evidences"]
        else:
            question = conversation.get_var("dataset").contents["X"].iloc[idx]["texts"]
            choices = conversation.get_var("dataset").contents["X"].iloc[idx]["choices"]

    return_s = f"Instance of ID <b>{idx}</b> <br>"

    _, pre_prediction = prediction_generation(data, conversation, idx, num_shot=3, given_first_field=None, given_second_field=None)

    aug = naw.SynonymAug(aug_src='wordnet')

    # Word augmenter
    if conversation.describe.get_dataset_name() == "covid_fact":
        # Augment both claim and evidence to create a new instance
        augmented_first_field = aug.augment(claim)
        augmented_second_field = aug.augment(evidence)

        return_s += f"<b>Claim</b>: {claim}<br>"
        return_s += f"<b>Original evidence:</b> {evidence}<br>"
        return_s += f"<b>Prediction before augmentation</b>: <span style=\"background-color: #6CB4EE\">{pre_prediction}</span><br><br>"
        return_s += f"<b>Augmented claim:</b> {augmented_first_field}<br>"
        return_s += f"<b>Augmented evidence:</b> {augmented_second_field}<br>"
    else:
        augmented_first_field = aug.augment(question)
        # augmented_first_field = get_augmentation(conversation, question)

        split_choices = choices.split("-")

        temp = []
        for i in split_choices:
            temp.append(aug.augment(i))

        return_s += f"<b>Original question:</b> {question}<br>"
        return_s += f"<b>Original choices:</b> {convert_str_to_options(choices)}<br>"
        return_s += f"<b>Prediction before augmentation</b>: <span style=\"background-color: #6CB4EE\">{split_choices[pre_prediction]}</span><br><br>"
        return_s += f"<b>Augmented question:</b> {augmented_first_field}<br>"

    _, post_prediction = prediction_generation(data, conversation, idx, num_shot=3, given_first_field=augmented_first_field, given_second_field=None)

    if conversation.describe.get_dataset_name() == "covid_fact":
        return_s += f"<b>Prediction after augmentation</b>: <span style=\"background-color: #6CB4EE\">{post_prediction}</span>"
    else:
        return_s += f"<b>Prediction after augmentation</b>: <span style=\"background-color: #6CB4EE\">{split_choices[post_prediction]}</span>"

    return return_s, 1
