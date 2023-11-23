import inseq

SUPPORTED_METHODS = ["integrated_gradients", "attention", "lime", "input_x_gradient"]


def handle_input(parse_text, i):
    """
    Handle the parse text and return the list of numbers(ids) and topk value if given
    Args:
        parse_text: parse_text from bot

    Returns: method, topk

    """
    if parse_text[i + 1] == "all":
        topk = 5
    else:
        try:
            topk = int(parse_text[i + 2])
        except ValueError:
            topk = 5
    try:
        if parse_text[i + 1] in SUPPORTED_METHODS:
            # Without topk
            method_name = parse_text[i + 1]
        elif parse_text[i + 2] in SUPPORTED_METHODS:
            # with all
            method_name = parse_text[i + 2]
        elif parse_text[i + 3] in SUPPORTED_METHODS:
            # with topk value
            method_name = parse_text[i + 3]
    except IndexError:
        method_name = "input_x_gradient"
    return topk, method_name


def feature_importance_operation(conversation, parse_text, i, **kwargs) -> (str, int):
    """
    feature attribution operation
    Args:
        conversation
        parse_text: parsed text from T5
        i: counter pointing at operation
        **kwargs:

    Returns:
        formatted string
    """
    # filter id 213 and nlpattribute all [E]
    # filter id 33 and nlpattribute topk 1 [E]

    # TODO: custom input

    topk, method_name = handle_input(parse_text, i)

    model = conversation.decoder.gpt_model

    inseq_model = inseq.load_model(
        model,
        method_name,
        device=str(conversation.decoder.gpt_model.device.type),  # Use same device as already loaded GPT model
    )

    # COVID-Fact dataset processing
    # TODO: Allow other datasets that don't have "evidences" and "claims"
    dataset = conversation.temp_dataset.contents["X"]
    evidences = dataset["evidences"].item()
    claims = dataset["claims"].item()

    # TODO: Import prompt from some central prompts file which are also used in prediction
    input_text = (f"Your task is to predict the veracity of the claim based on the evidence. \n"
                  f"Evidence: '{evidences}' \n"
                  f"Claim: '{claims}' \n"
                  f"Please provide your answer as one of the labels: Refuted or Supported. \n"
                  f"Veracity prediction: ")
    tokenized_input_text = conversation.decoder.gpt_tokenizer(input_text)
    tokenized_length = len(tokenized_input_text.encodings[0].ids)

    # Attribute text
    out = inseq_model.attribute(
        input_texts=input_text,
        n_steps=1,
        return_convergence_delta=True,
        step_scores=["probability"],  # TODO: Check if necessary
        show_progress=True,  # TODO: Check if necessary
        generation_args={"max_length": tokenized_length + 5},  # Dirty solution: Constrain to 5 new tokens
    )

    out_agg = out.aggregate(inseq.data.aggregator.SubwordAggregator)

    # TODO: Check if "Supported" or "Refuted" is the first token
    # Extract 1D heatmap (attributions for first token)
    final_agg = out_agg[0].aggregate()
    first_token_attributions = final_agg.target_attributions[:, 0]

    # TODO: Possibly reduce to tokens in "claim" and "evidence" (exclude the prompt)

    # Get HTML visualization from Inseq
    heatmap_viz = out_agg.show(return_html=True).split("<html>")[1].split("</html>")[0]

    def k_highest_indices(lst, k):
        # Create a list of tuples (value, index)
        indexed_lst = list(enumerate(lst))
        # Sort the list by the values in descending order
        sorted_lst = sorted(indexed_lst, key=lambda x: x[1], reverse=True)
        # Extract the first k indices
        highest_indices = [index for index, value in sorted_lst[:k]]
        return highest_indices

    topk_tokens = [final_agg.target[i].token for i in k_highest_indices(first_token_attributions, topk)]

    # TODO: Find sensible verbalization
    return_s = f"Top {topk} token(s):<br>"
    for i in topk_tokens:
        if i == "<s>":  # This token causes strikethrough text in HTML! 🤨
            i = "< s >"
        return_s += f"<b>{i}</b><br>"

    return_s += "<details><summary>"
    return_s += "The visualization: "
    return_s += "</summary>"
    return_s += heatmap_viz
    return_s += "</details><br>"

    return return_s, 1
