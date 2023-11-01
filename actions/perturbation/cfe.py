from actions.prediction.predict import prediction_generation


def counterfactuals_operation(conversation, parse_text, i, **kwargs):
    idx = None
    data = None

    if conversation.custom_input is not None and conversation.used is False:
        if conversation.describe.get_dataset_name() == "covid_fact":
            # TODO
            claim, evidence = conversation.custom_input, conversation.custom_input
        else:
            # TODO
            pass
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
            # TODO
            pass

    model = conversation.decoder.gpt_model
    tokenizer = conversation.decoder.gpt_tokenizer

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Get prediction
    _, prediction = prediction_generation(data, conversation, idx)

    prompt_template = ""

    if conversation.describe.get_dataset_name() == "covid_fact":

        prompt_template += f"claim: {claim}"
        prompt_template += f"evidence: {evidence}"

        if prediction == "SUPPORTED":
            reversed_prediction = "REFUTED"
        else:
            reversed_prediction = "SUPPORTED"

        # zero-shot prompting
        prompt_template += f"Based on evidence, the claim is {prediction.lower()}. Modify only the claim such that " \
                           f"the claim becomes {reversed_prediction.lower()} based on evidence."
    else:
        # TODO
        pass

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=128)
    result = tokenizer.decode(output[0]).split(prompt_template)[1][:-4]

    return_s = ""

    if conversation.describe.get_dataset_name() == "covid_fact":
        return_s += f"Instance with ID <b>{idx}</b>"
        return_s += f"<b>Original claim:</b> {claim}<br>"
        return_s += f"<b>Counterfactual:</b> {result} <br>"
        return_s += f"<b>Evidence: </b> {evidence}"
        return_s += "<br><br>"
    else:
        # TODO
        pass


    return return_s, 1
