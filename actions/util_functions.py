"""Util functions."""
import numpy as np
from sklearn.tree import _tree

from logic.conversation import Conversation


def gen_parse_op_text(conversation):
    """Generates a piece of text summarizing the parse operation.

    Note that the first term in the parse op list is supposed to be an and or or
    which is stripped off here to make formatting that list in the operations easier.
    """
    ret_text = ""
    conv_parse_ops = conversation.parse_operation
    for i in range(1, len(conv_parse_ops)):
        ret_text += conv_parse_ops[i] + " "
    ret_text = ret_text[:-1]
    return ret_text


def convert_categorical_bools(data):
    if data == 'true':
        return 1
    elif data == 'false':
        return 0
    else:
        return data


def get_parse_filter_text(conversation: Conversation):
    """Gets the starting parse text."""
    parse_op = gen_parse_op_text(conversation)
    temp_dataset_size = len(conversation.temp_dataset.contents["X"])
    full_dataset_size = len(conversation.stored_vars["dataset"].contents["X"])

    if len(parse_op) > 0:
        intro_text = f"For the data with <b>{parse_op}</b>,"
    elif temp_dataset_size < full_dataset_size:
        last_filter_candidate = None
        subset_size_str = f"({temp_dataset_size} out of {full_dataset_size}),"
        for parse in conversation.last_parse_string:
            if "includes '" in parse:
                last_filter_candidate = f"For the data including <b>{conversation.include_word}</b> {subset_size_str}"
            # TODO: Account for other types of filters, e.g. predictionfilter and labelfilter
        if not last_filter_candidate:
            last_filter_candidate = f"For this <b>subset</b> of the data {subset_size_str}"
        intro_text = last_filter_candidate
    else:
        intro_text = "For <b>all</b> the instances in the data,"
    return intro_text


def get_rules(tree, feature_names, class_names):
    # modified from https://mljar.com/blog/extract-rules-decision-tree/
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        incorrect_class = False

        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += "<b>" + str(p) + "</b>"
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            largest = np.argmax(classes)
            if class_names[largest] == "incorrect":
                incorrect_class = True
            rule += f"then the model is incorrect <em>{np.round(100.0 * classes[largest] / np.sum(classes), 2)}%</em>"
        rule += f" over <em>{path[-1][1]:,}</em> samples"

        if incorrect_class:
            rules += [rule]
    return rules
