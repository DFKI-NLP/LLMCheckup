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

    y_true, y_pred, ids = get_predictions_and_labels(conversation)

    if metric == "default" or metric == 'accuracy':
        metric = conversation.default_metric

    data_name = get_parse_filter_text(conversation).replace('For ', '')
    multi_class = True if dataset_name == 'ECQA' else False
    text = conversation.describe.get_score_text(y_true,
                                                y_pred,
                                                metric,
                                                conversation.rounding_precision,
                                                data_name,
                                                multi_class,
                                                average)

    text += "<br><br>"
    return text, 1
