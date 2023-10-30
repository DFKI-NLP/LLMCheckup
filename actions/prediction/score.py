"""Score operation.

This operation computes a score metric on the data or the eval data.
"""
import json

import numpy as np

from actions.prediction.pred_utils import get_predictions_and_labels
from actions.util_functions import get_parse_filter_text
from timeout import timeout

MAPPING = {'dummy': 0, 'inform': 1, 'question': 2, 'directive': 3, 'commissive': 4}


@timeout(60)
def score_operation(conversation, parse_text, i, **kwargs):
    """Self description."""

    # Get the name of the metric
    metric = parse_text[i + 1]

    # Get the dataset name
    dataset_name = conversation.describe.get_dataset_name()

    average = None
    # if dataset_name == "daily_dialog":
    #     flags = ["micro", "macro", "weighted"]
    #     try:
    #         average = parse_text[i + 2]
    #     except ValueError:
    #         pass
    #     except IndexError:
    #         pass
    #     if metric not in ["default", "accuracy", "roc"]:
    #         if len(parse_text) == 2 or parse_text[i + 2] == '[e]':
    #             average = "macro"
    #         elif parse_text[i+2] in flags:
    #             average = parse_text[i+2]
    #         else:
    #             raise NotImplementedError(f"Flag {average} is not supported!")

    data_indices = conversation.temp_dataset.contents["X"].index.to_list()
    print(len(data_indices))
    y_true, y_pred, ids = get_predictions_and_labels(dataset_name, data_indices, conversation)

    if metric == "default" or metric == 'accuracy':
        metric = conversation.default_metric

    # if dataset_name == 'daily_dialog' and metric == "roc":
    #     path = f"./cache/{dataset_name}/ig_explainer_{dataset_name}_prediction.json"
    #
    #     fileObject = open(path, "r")
    #     jsonContent = fileObject.read()
    #     json_list = json.loads(jsonContent)
    #
    #     y_pred = []
    #     for item in json_list:
    #         if item["batch"] in ids:
    #             y_pred.append(item["predictions"])
    #     y_pred = np.array(y_pred)

    data_name = get_parse_filter_text(conversation).replace('For ', '')
    multi_class = True if dataset_name == 'daily_dialog' else False
    text = conversation.describe.get_score_text(y_true,
                                                y_pred,
                                                metric,
                                                conversation.rounding_precision,
                                                data_name,
                                                multi_class,
                                                average)

    text += "<br><br>"
    return text, 1
