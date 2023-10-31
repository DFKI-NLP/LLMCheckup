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


def feature_importance_operation(conversation, parse_text, i, **kwargs):
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

    id_list, topk = handle_input(parse_text)

    if topk is None:
        topk = 5

    model = conversation.decoder.gpt_model
    tokenizer = conversation.decoder.gpt_tokenizer

    model_name = conversation.decoder.parser_name
    if model_name == 'EleutherAI/gpt-neo-2.7B':
        fileObject = open("./cache/gpt-neo-2.7b_feature_attribution.json", "r")
    elif model_name == "EleutherAI/gpt-j-6b":
        fileObject = open("./cache/gpt-j-6b_feature_attribution.json", "r")
    else:
        raise NotImplementedError(f"Model {model_name} is unknown!")

    jsonContent = fileObject.read()
    json_list = json.loads(jsonContent)

    return_s = ""
    for _id in id_list:
        attribution = json_list[_id]["attribution"]
        texts = json_list[_id]["text"]
        decoded_text = [tokenizer.convert_tokens_to_string(text) for text in texts]
        return_s += get_visualization(attribution, topk, decoded_text)
        return_s += "<br><br>"
    return return_s, 1
