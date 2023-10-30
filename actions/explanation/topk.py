import os
import json
import torch
from tqdm import tqdm


def results_with_pattern(results, reverse):
    """
    Output the results with certain pattern

    Args:
        results: attribution scores list
    Returns:
        None
    """
    # example: dumb, fucking, and ugly are the most attributed for the hate speech label
    if len(results) == 1:
        if reverse:
            return results[0][0] + " is the most attributed"
        else:
            return results[0][0] + " is the least attributed"
    else:
        string = ""
        for i in range(len(results) - 1):
            string += results[i][0] + ", "
        string += "and "
        string += results[len(results) - 1][0]
        if not reverse:
            return string + " are the least attributed."
        else:
            return string + " are the most attributed."


def topk(conversation, k, class_idx=None, reverse=True):
    """
    The operation to get most k important tokens

    Args:
        conversation: conversation object
        k (int): number of tokens
        class_idx: filter label (index)
        reverse: ordering
    Returns:
        sorted_scores: top k important tokens
    """
    model_name = conversation.decoder.parser_name
    if model_name == 'EleutherAI/gpt-neo-2.7B':
        res_path = f"./cache/olid_gpt-neo-2.7b_globaltopk.json"
        data_path = "./cache/gpt-neo-2.7b_feature_attribution.json"
        pred_path = f"./cache/guided/olid_gpt-neo-2.7b_5_shot_prediction.json"
    elif model_name == "EleutherAI/gpt-j-6b":
        res_path = f"./cache/olid_gpt-j-6b_globaltopk.json"
        data_path = "./cache/gpt-j-6b_feature_attribution.json"
        pred_path = f"./cache/guided/olid_gpt-j-6b_20_shot_prediction.json"
    else:
        raise NotImplementedError(f"Model {model_name} is unknown!")

    tokenizer = conversation.decoder.gpt_tokenizer

    if os.path.exists(res_path) and (class_idx is None):
        fileObject = open(res_path, "r")
        jsonContent = fileObject.read()
        result_list = json.loads(jsonContent)

        if len(result_list) >= k:
            if not reverse:
                return results_with_pattern(result_list[::-1][:k], reverse=True)
            else:
                return results_with_pattern(result_list[:k], reverse=True)
        else:
            print("[Info] The length of score is smaller than k")
            if not reverse:
                return results_with_pattern(result_list[::-1], reverse=True)
            else:
                return results_with_pattern(result_list, reverse=True)

    fileObject = open(data_path, "r")
    jsonContent = fileObject.read()
    results = json.loads(jsonContent)

    # individual tokens
    word_set = set()
    word_counter = {}
    word_attributions = {}

    if class_idx:
        #print('class name: ', class_idx)
        temp = []

        fileObject = open(pred_path, "r")
        jsonContent = fileObject.read()
        pred = json.loads(jsonContent)

        for res in pred:
            if res["prediction"] == class_idx:
                temp.append(results[res["idx"]])
    else:
        temp = results

    pbar = tqdm(temp)

    for result in pbar:
        pbar.set_description('Processing Attribution')
        attribution = result["attribution"]
        tokens = [(tokenizer.convert_tokens_to_string(t)).strip() for t in result["text"]]
        counter = 0

        # count for attributions and #occurance
        for token in tokens:
            if not token in word_set:
                word_set.add(token)
                word_counter[token] = 1
                word_attributions[token] = attribution[counter]
            else:
                word_counter[token] += 1
                word_attributions[token] += attribution[counter]
            counter += 1

    scores = {}
    for word in word_set:
        scores[word] = (word_attributions[word] / word_counter[word])

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=reverse)

    if not os.path.exists(res_path) and class_idx is None:
        jsonString = json.dumps(sorted_scores)
        jsonFile = open(res_path, "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    if len(sorted_scores) >= k:
        return results_with_pattern(sorted_scores[:k], reverse=True)
    else:
        return results_with_pattern(sorted_scores, reverse=True)