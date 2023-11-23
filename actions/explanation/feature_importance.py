import inseq
import json
import numpy as np


def handle_input(parse_text):
    """
    Handle the parse text and return the list of numbers(ids) and topk value if given
    Args:
        parse_text: parse_text from bot

    Returns: id_list, topk

    """
    id_list = []
    topk = None

    for item in parse_text:
        try:
            if int(item):
                if int(item) > 0:
                    id_list.append(int(item))
        except:
            pass

    if "topk" in parse_text:
        if len(id_list) >= 1:
            topk = id_list[-1]

        # filter id 5 or filter id 151 or filter id 315 and nlpattribute topk 10 [E]
        if len(id_list) > 1:
            return id_list[:-1], topk
        else:
            # nlpattribute topk 3 [E]
            return None, topk
    else:
        if len(id_list) >= 1:
            # filter id 213 and nlpattribute all [E]
            if "all" in parse_text:
                return id_list, None

        # nlpattribute [E]
        return id_list, topk


def get_visualization(attr, topk, original_text):
    """
    Get visualization on given input
    Args:
        attr: attribution list
        topk: top k value
        original_text: original text
        conversation: conversation object

    Returns:
        heatmap in html form
    """
    return_s = ""

    # Get indices according to absolute attribution scores ascending
    idx = np.argsort(np.absolute(np.copy(attr)))

    # Get topk tokens
    topk_tokens = []
    for i in np.argsort(attr)[-topk:][::-1]:
        topk_tokens.append(original_text[i])

    score_ranking = []
    for i in range(len(idx)):
        score_ranking.append(list(idx).index(i))
    fraction = 1.0 / (len(original_text) - 1)

    return_s += f"Top {topk} token(s): "
    for i in topk_tokens:
        return_s += f"<b>{i}</b>"
        return_s += " "
    return_s += '<br>'

    return_s += "<details><summary>"
    return_s += "The visualization: "
    return_s += "</summary>"
    # for i in range(1, len(text_list) - 1):
    for i in range(len(original_text)):
        if attr[i] >= 0:
            # Assign red to tokens with positive attribution
            return_s += f"<span style='background-color:rgba(255,0,0,{round(fraction * score_ranking[i], 2)});padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 2; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone'>"
        else:
            # Assign blue to tokens with negative attribution
            return_s += f"<span style='background-color:rgba(0,0,255,{round(fraction * score_ranking[i], 2)});padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 2; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone'>"
        # return_s += text_list[i]
        return_s += original_text[i]
        return_s += "</span>"
        return_s += ' '
    return_s += "</details>"
    return_s += '<br><br><br>'

    return return_s


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
    # filter id 5 or filter id 151 or filter id 315 and nlpattribute topk 10 [E]
    # filter id 213 and nlpattribute all [E]
    # filter id 33 and nlpattribute topk 1 [E]

    # TODO: handle "all" and "topk" cases

    # TODO: custom input

    id_list, topk = handle_input(parse_text)

    if topk is None:
        topk = 5  # TODO: Currently unused

    model = conversation.decoder.gpt_model

    inseq_model = inseq.load_model(
        model,
        "input_x_gradient",  # TODO: Allow different choices of attribution_method
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

