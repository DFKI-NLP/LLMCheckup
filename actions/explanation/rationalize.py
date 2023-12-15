import torch

from actions.prediction.predict import prediction_generation, convert_str_to_options
from actions.prompt_type import type2prompt

from actions.util_functions import gen_parse_op_text


def rationalize_operation(conversation, parse_text, i, **kwargs):
    id_list = []
    for item in parse_text:
        try:
            if type(int(item)) == int:
                id_list.append(int(item))
        except ValueError:
            pass

    model = conversation.decoder.gpt_model
    tokenizer = conversation.decoder.gpt_tokenizer

    return_s = ""
    prompt_template = ""

    data = conversation.temp_dataset.contents['X']

    # Get claims and evidences
    if conversation.describe.get_dataset_name() == "covid_fact":
        if conversation.custom_input is None and conversation.used is False:
            claim, evidence = conversation.custom_input['first_input'], conversation.custom_input['second_input']
            _, prediction = prediction_generation(None, conversation, None)
        else:
            for i, feature_name in enumerate(data.columns):
                if feature_name == "claims":
                    claim = data[feature_name].values[0]
                elif feature_name == "evidences":
                    evidence = data[feature_name].values[0]

            _, prediction = prediction_generation(data, conversation, id_list[0])

        prompt_template += f"claim: {claim}"
        prompt_template += f"evidence: {evidence}"

        # zero-shot prompting
        prompt_template += f"Based on evidence, the prediction of the claim is {prediction.lower()}. Explain why it " \
                           f"is predicted as {prediction.lower()}."
    else:
        if conversation.custom_input is None and conversation.used is False:
            question, choices = conversation.custom_input['first_input'], conversation.custom_input['second_input']
            _, prediction = prediction_generation(None, conversation, None)
        else:
            for i, feature_name in enumerate(data.columns):
                if feature_name == "texts":
                    texts = data[feature_name].values[0]
                elif feature_name == "choices":
                    choices = data[feature_name].values[0]

            _, prediction = prediction_generation(data, conversation, id_list[0])

        prediction = choices.split("-")[prediction]
        prompt_template += f"text: {texts}"
        prompt_template += f"choice: {convert_str_to_options(choices)}."

        # zero-shot prompting
        prompt_template += f"Based on text, the prediction of the choice is {prediction}. Explain why it " \
                           f"is predicted as {prediction}."

    # Append additional prompts from user
    if conversation.prompt_type != "none":
        if conversation.prompt_type in type2prompt.keys():
            prompt_template += type2prompt[conversation.prompt_type]
        else:
            prompt_template += conversation.prompt_type
    print(f"[Prompt] Using customized additional prompt: *** {conversation.prompt_type} ***")

    conversation.current_prompt = prompt_template

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.to(model.device.type)
    with torch.no_grad():
        output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    result = tokenizer.decode(output[0]).split(prompt_template)[1][:-4]

    filter_string = gen_parse_op_text(conversation)
    return_s += f"The instance with <b>{filter_string}</b>: <br>"

    if conversation.describe.get_dataset_name() == "covid_fact":
        return_s += f"<b>Claim</b>: {claim}<br>"
        return_s += f"<b>Evidence</b>: {evidence}<br>"
    else:
        return_s += f"<b>Text</b>: {texts}<br>"
        return_s += f"<b>Choices</b>: {convert_str_to_options(choices)}<br>"

    return_s += f"The <b>prediction</b> is <span style=\"background-color: #6CB4EE\">{prediction}</span>.<br>"
    return_s += f"<b>Reasoning: </b><br>"
    return_s += "<details>"

    return_s += f"<summary>{result[:200]}...</summary>"
    return_s += result

    return_s += "</details>"

    return return_s, 1
