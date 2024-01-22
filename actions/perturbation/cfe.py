import torch

from actions.demonstration_utils import get_cfe_prompt_by_demonstrations
from actions.prediction.predict import prediction_generation, convert_str_to_options


def counterfactuals_operation(conversation, parse_text, i, **kwargs):
    idx = None
    data = None

    if conversation.custom_input is not None and conversation.used is False:
        if conversation.describe.get_dataset_name() == "covid_fact":
            claim, evidence = conversation.custom_input["first_input"], conversation.custom_input["second_input"]
        else:
            question, choices = conversation.custom_input['first_input'], conversation.custom_input['second_input']
    else:
        assert len(conversation.temp_dataset.contents["X"]) == 1

        data = conversation.temp_dataset.contents['X']

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

    model = conversation.decoder.gpt_model
    tokenizer = conversation.decoder.gpt_tokenizer

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Get prediction
    _, prediction = prediction_generation(data, conversation, idx)

    prompt_template = ""

    if conversation.describe.get_dataset_name() == "covid_fact":

        prompt_template += get_cfe_prompt_by_demonstrations("covid_fact", claim, evidence, prediction)
    else:
        prompt_template += get_cfe_prompt_by_demonstrations("ecqa", question, choices, int(prediction))

    conversation.current_prompt = prompt_template

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.to(model.device.type)
    with torch.no_grad():
        output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=128)
    result = tokenizer.decode(output[0]).split(prompt_template)[1][:-4]

    return_s = f"Instance with ID <b>{idx}</b><br>"

    if conversation.describe.get_dataset_name() == "covid_fact":
        _, post_prediction = prediction_generation(data, conversation, idx, given_first_field=None,
                                                   given_second_field=result)
        return_s += f"<b>Original claim:</b> {claim}<br>"
        return_s += f"<b>Evidence: </b> {evidence}<br>"
        return_s += f"<b>Prediction</b> <span style=\"background-color: #6CB4EE\">{prediction}</span><br>"
        return_s += f"<b>Counterfactual of evidence:</b> {result} <br>"
        return_s += f"<b>Prediction of counterfactual:</b> <span style=\"background-color: #6CB4EE\">{post_prediction}</span>"
        return_s += "<br><br>"
    else:
        _, post_prediction = prediction_generation(data, conversation, idx, given_first_field=result,
                                                   given_second_field=None)
        return_s += f"<b>Original question:</b> {question}<br>"
        return_s += f"<b>Choices: </b> {convert_str_to_options(choices)}<br>"
        return_s += f"<b>Prediction</b> <span style=\"background-color: #6CB4EE\">{choices.split('-')[prediction]}</span><br>"
        return_s += f"<b>Counterfactual of question:</b> {result} <br>"
        return_s += f"<b>Prediction of counterfactual:</b> <span style=\"background-color: #6CB4EE\">{choices.split('-')[post_prediction]}</span>"
        return_s += "<br><br>"

    return return_s, 1
