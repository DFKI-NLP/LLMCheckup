"""Data augmentation operation."""

import nlpaug.augmenter.word as naw
import torch
from actions.demonstration_utils import get_augmentation_prompt_by_demonstrations
from actions.prediction.predict import convert_str_to_options, prediction_generation


def get_augmentation(conversation, field, num_token=64):
    """Get augmented text based on original text.

    :param conversation: conversation object
    :param field: input text
    :param num_token: number of max token
    :return: augmented text
    """
    model = conversation.decoder.gpt_model
    tokenizer = conversation.decoder.gpt_tokenizer

    prompt_template = prompt_template = (
        "Paraphrase the following text. Please, do not change its meaning and do not add any additional information.\nOriginal text: "
        + field
        + "\nParaphrased text:"
    )
    if conversation.describe.get_dataset_name() == "ECQA":
        prompt_template = (
            "Do not answer any questions, just paraprase the text. Keep the question as question.\nOriginal question: "
            + field
            + "\nParaphrased question:"
        )
    conversation.current_prompt = prompt_template

    input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids.to(model.device.type)
    with torch.no_grad():
        output = model.generate(
            inputs=input_ids,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=40,
            max_new_tokens=num_token,
        )
    result = tokenizer.decode(output[0]).split(prompt_template)[1][:-4]
    if "\n" in result:
        result = result[: result.index("\n") + 1]

    return result.strip()


def get_sample_augmentation(data, conversation, idx, first_field, second_field, aug=None):
    """Sample-wise data augmentation.

    :param data: current dataset
    :param conversation: conversation object
    :param idx: sample ID
    :param first_field: text of the first field (claim for COVID-Fact, question for ECQA)
    :param second_field: text of the second field (evidence for ECQA)
    :param aug: NLPAug word-level augmenter or None if we use LLM prompting
    :return: augmented fields and model prediction with the new (augmented) inputs
    """
    augmented_first_field = None
    augmented_second_field = None
    if not conversation.llm_augmentation:
        assert aug is not None
        augmented_first_field = aug.augment(first_field)
        if second_field is not None:
            augmented_second_field = aug.augment(second_field)
    else:
        augmented_first_field = get_augmentation(conversation, field=first_field, num_token=64)
        if second_field is not None:
            augmented_second_field = get_augmentation(
                conversation, field=second_field, num_token=256
            )

    _, post_prediction = prediction_generation(
        data,
        conversation,
        idx,
        num_shot=3,
        given_first_field=augmented_first_field,
        given_second_field=augmented_second_field,
    )

    return post_prediction, augmented_first_field, augmented_second_field


def augment_operation(conversation, parse_text, i, **kwargs):
    """Data augmentation.

    :param conversation: conversation object
    :param parse_text: parsed operation text (not used)
    :param i: current index
    :return: augmented text
    """
    data = conversation.temp_dataset.contents["X"]
    idx = None

    if conversation.custom_input is not None and conversation.used is False:
        if conversation.describe.get_dataset_name() == "covid_fact":
            claim, evidence = (
                conversation.custom_input["first_input"],
                conversation.custom_input["second_input"],
            )
            first_field = claim
            second_field = evidence
        else:
            question, choices = (
                conversation.custom_input["first_input"],
                conversation.custom_input["second_input"],
            )
            first_field = question
            second_field = None
    else:
        assert len(conversation.temp_dataset.contents["X"]) == 1

        try:
            idx = conversation.temp_dataset.contents["X"].index[0]
        except ValueError:
            return "Sorry, invalid id", 1

        if conversation.describe.get_dataset_name() == "covid_fact":
            claim = conversation.get_var("dataset").contents["X"].iloc[idx]["claims"]
            evidence = conversation.get_var("dataset").contents["X"].iloc[idx]["evidences"]
            first_field = claim
            second_field = evidence
        else:
            question = conversation.get_var("dataset").contents["X"].iloc[idx]["texts"]
            choices = conversation.get_var("dataset").contents["X"].iloc[idx]["choices"]
            first_field = question
            second_field = None

    return_s = f"Instance of ID <b>{idx}</b> <br>"

    _, pre_prediction = prediction_generation(
        data,
        conversation,
        idx,
        num_shot=3,
        given_first_field=first_field,
        given_second_field=second_field,
    )

    if not conversation.llm_augmentation:
        aug = naw.SynonymAug(aug_src="wordnet")
    else:
        aug = None
    patience = 2  # how many times we repeat data augmentation to get the output that differs from the original input
    count_prediction = 0

    post_prediction, augmented_first_field, augmented_second_field = get_sample_augmentation(
        data, conversation, idx, first_field, second_field, aug
    )
    # repeat data augmentation if prediction changes the label
    while count_prediction < patience and post_prediction != pre_prediction:
        count_prediction += 1
        post_prediction, augmented_first_field, augmented_second_field = get_sample_augmentation(
            data, conversation, idx, first_field, second_field, aug
        )
    # build the return string for the interface
    if conversation.describe.get_dataset_name() == "covid_fact":
        return_s += f"<b>Claim</b>: {claim}<br>"
        return_s += f"<b>Original evidence:</b> {evidence}<br>"
        return_s += (
            f'<b>Prediction before augmentation</b>: <span style="background-color: #6CB4EE">'
            f"{pre_prediction}</span><br><br>"
        )
        return_s += f"<b>Augmented claim:</b> {augmented_first_field}<br>"
        return_s += f"<b>Augmented evidence:</b> {augmented_second_field}<br>"
        return_s += (
            f'<b>Prediction after augmentation</b>: <span style="background-color: #6CB4EE">'
            f"{post_prediction}</span>"
        )
    else:
        split_choices = choices.split("-")
        return_s += f"<b>Original question:</b> {question}<br>"
        return_s += f"<b>Original choices:</b> {convert_str_to_options(choices)}<br>"
        return_s += (
            f'<b>Prediction before augmentation</b>: <span style="background-color: #6CB4EE">'
            f"{split_choices[pre_prediction]}</span><br><br>"
        )
        return_s += f"<b>Augmented question:</b> {augmented_first_field}<br>"
        return_s += (
            f'<b>Prediction after augmentation</b>: <span style="background-color: #6CB4EE">'
            f"{split_choices[post_prediction]}</span>"
        )

    return return_s, 1
