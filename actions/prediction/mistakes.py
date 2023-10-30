"""Show model mistakes"""
import gin
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from actions.util_functions import get_parse_filter_text, get_rules
from actions.prediction.pred_utils import get_predictions_and_labels
from timeout import timeout

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def one_mistake(y_true, y_pred, conversation, intro_text):
    """One mistake text"""
    label = y_true[0]
    prediction = y_pred[0]

    label_text = conversation.get_class_name_from_label(label)
    predict_text = conversation.get_class_name_from_label(prediction)

    if label == prediction:
        correct_text = "correct"
    else:
        correct_text = "incorrect"

    return_string = (f"{intro_text} the model predicts <em>{predict_text}</em> and the ground"
                     f" label is <em>{label_text}</em>, so the model is <b>{correct_text}</b>!")
    return return_string


def sample_mistakes(y_true, y_pred, conversation, intro_text, ids):
    """Sample mistakes sub-operation
    `mistake sample [E]`
    """
    if len(y_true) == 1:
        return_string = one_mistake(y_true, y_pred, conversation, intro_text)
    else:
        incorrect_num = np.sum(y_true != y_pred)
        total_num = len(y_true)
        incorrect_data = ids[y_true != y_pred]
        incorrect_str = "<details><summary>Here are the ids of instances the model predicts incorrectly:</summary>" \
                        f"{', '.join([str(d) for d in incorrect_data])}</details>"

        error_rate = round(incorrect_num / total_num, conversation.rounding_precision)
        return_string = (f"{intro_text} the model is incorrect {incorrect_num} out of {total_num} "
                         f"times (error rate {error_rate}). <br>{incorrect_str}")

    return return_string


def count_mistakes(y_true, y_pred, conversation, intro_text):
    """"Count number of instances that are predicted wrongly """
    incorrect_num = np.sum(y_true != y_pred)
    total_num = len(y_true)
    error_rate = round(incorrect_num / total_num, conversation.rounding_precision)

    return_string = (f"{intro_text} the model is incorrect {incorrect_num} out of {total_num} "
                     f"times (error rate {error_rate}).")

    labels = [v for k,v in conversation.class_names.items()]
    y_true_labels = [conversation.get_class_name_from_label(lbl) for lbl in y_true]
    y_pred_labels = [conversation.get_class_name_from_label(lbl) for lbl in y_pred]
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    try:
        disp.plot()
        if not os.path.exists('./static/plots'):
            os.mkdir('./static/plots')
        plt.savefig('./static/plots/confusion_matrix.png')
        image_tag = "<img src='./static/plots/confusion_matrix.png' style='width:420px;height:350px;'>"
        return_string += "<br>Here is the confusion matrix:<br> " + image_tag
    except ValueError:
        pass

    return return_string


def train_tree(data, target, depth: int = 1):
    """Trains a decision tree"""
    dt_string = []
    tries = 0
    while len(dt_string) < 3 and tries < 10:
        tries += 1
        dt = DecisionTreeClassifier(max_depth=depth).fit(data, target)
        dt_string = get_rules(dt,
                              feature_names=list(data.columns),
                              class_names=["correct", "incorrect"])
        depth += 1

    return dt_string


def typical_mistakes(data, y_true, y_pred, conversation, intro_text, ids):
    """Typical mistakes sub-operation
    `mistake typical [E]`
    """
    if len(y_true) == 1:
        return_string = one_mistake(y_true, y_pred, conversation, intro_text)
    else:
        incorrect_vals = y_true != y_pred
        return_options = train_tree(data, incorrect_vals)

        if len(return_options) == 0:
            return "I couldn't find any patterns for mistakes the model typically makes."

        return_string = f"{intro_text} the model typically predicts incorrect:<br><br>"
        for rule in return_options:
            return_string += rule + "<br><br>"

    return return_string


@timeout(60)
@gin.configurable
def show_mistakes_operation(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows the model mistakes."""

    # Get dataset name
    name = conversation.describe.get_dataset_name()
    data_indices = conversation.temp_dataset.contents["X"].index.to_list()
    y_pred, y_true, ids = get_predictions_and_labels(name, data_indices)

    # The filtering text
    intro_text = get_parse_filter_text(conversation)

    if len(y_true) == 0:
        return "There are no instances in the data that meet this description.<br><br>", 0

    if np.sum(y_true == y_pred) == len(y_true):
        if len(y_true) == 1:
            return f"{intro_text} the model predicts correctly!<br><br>", 1
        else:
            return f"{intro_text} the model predicts correctly on all the instances in the data!<br><br>", 1

    if parse_text[i + 1] == "sample":
        return_string = sample_mistakes(y_true,
                                        y_pred,
                                        conversation,
                                        intro_text,
                                        ids)
    elif parse_text[i + 1] == "count":
        return_string = count_mistakes(y_true,
                                       y_pred,
                                       conversation,
                                       intro_text)
    # elif parse_text[i+1] == "typical":
    #     return_string = typical_mistakes(data,
    #                                      y_true,
    #                                      y_pred,
    #                                      conversation,
    #                                      intro_text,
    #                                      ids)
    else:
        raise NotImplementedError(f"No mistake type {parse_text[i + 1]}")

    return_string += "<br><br>"
    return return_string, 1
