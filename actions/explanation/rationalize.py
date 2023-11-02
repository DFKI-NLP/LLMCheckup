from actions.prediction.predict import prediction_generation

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
                           f"is predicted as {prediction.lower()}. Let's think step by step."
    else:
        # TODO
        pass

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    result = tokenizer.decode(output[0]).split(prompt_template)[1][:-4]

    filter_string = gen_parse_op_text(conversation)
    return_s += f"The instance with <b>{filter_string}</b>: <br>"

    if conversation.describe.get_dataset_name() == "covid_fact":
        return_s += f"<b>Claim</b>: {claim}<br>"
        return_s += f"<b>Evidence</b>: {evidence}<br>"
    else:
        # TODO
        pass

    return_s += f"The prediction is {prediction}.<br>"
    return_s += f"<b>Reasoning: </b><br>"
    return_s += result

    return return_s, 1
