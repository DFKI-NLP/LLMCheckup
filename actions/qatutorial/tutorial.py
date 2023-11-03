def tutorial_operation(conversation, parse_text, i, **kwargs):
    level = conversation.qa_level

    return_s = ""
    if level == "beginner":
        user_input = conversation.user_input
        prompt_template = f"As a beginner in NLP, could you answer me the following question in details: {user_input}"
        print(prompt_template)

        model = conversation.decoder.gpt_model
        tokenizer = conversation.decoder.gpt_tokenizer

        input_ids = tokenizer(prompt_template, return_tensors='pt')
        output = model.generate(**input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40,
                                max_new_tokens=128)
        result = tokenizer.decode(output[0]).split(prompt_template)[1][:-4]

        return_s += result

    else:
        # For people with expertise and expert, just return tooltips
        ops = parse_text[i + 1]

        if ops == "qafa":
            return_s += "Indicates which <b>tokens</b> for a <b>single example</b> are <b>most important</b>. " \
                        "Also prints a heatmap visualization."
        elif ops == "qada":
            return_s += "Generates a <b>modified</b> version of a given <b>single example</b> that can be used as a " \
                        "<b>new data point</b>."
        elif ops == "qasim":
            return_s += "Retrieves a number of <b>examples</b> from the dataset that are semantically <b>similar</b> " \
                        "to a given <b>single example</b>."
        elif ops == "qacfe":
            return_s += "Generates a <b>modified</b> version of a given <b>single example</b> that <b>flips</b> the " \
                        "model's <b>prediction</b> from one label to another."
        elif ops == "qarationale":
            return_s += "Generates a <b>free-text justification</b> for the model's prediction on a <b>single " \
                        "example</b>."
        else:
            raise NotImplementedError(f"Operation {ops} is not supported!")

    return return_s, 1
