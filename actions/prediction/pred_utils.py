import json

import numpy as np
import pandas as pd

from logic.utils import read_precomputed_prediction

str2int = {"offensive": 1, "non-offensive": 0}


def get_predictions_and_labels(name, indices, conversation):
    """
    Args:
        name: dataset name
        indices: indices of temp_dataset
    Returns:
        predictions and labels
    """
    json_list = read_precomputed_prediction(conversation)
    y_pred, y_true, ids = [], [], []

    for item in json_list:
        if item["idx"] in indices:
            y_pred.append(str2int[item["prediction"]])
            ids.append(item["idx"])

    df = pd.read_csv("./data/offensive_val.csv")
    labels = list(df["label"])
    for i in range(len(labels)):
        if i in indices:
            y_true.append(labels[i])

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    id_array = np.array(ids)

    return y_pred, y_true, id_array
