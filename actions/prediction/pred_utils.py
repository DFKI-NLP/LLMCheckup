import numpy as np


def get_predictions_and_labels(conversation):
    """
    Get predictions and labels from subset
    :param conversation: conversation objects
    :return: predictions, golden labels, indices
    """
    df = conversation.precomputation_of_prediction

    y_pred = []
    y_true = []
    ids = []

    for i in range(len(df)):
        ids.append(df.loc[i]["id"])
        y_pred.append(df.loc[i]["prediction"])
        y_true.append(conversation.temp_dataset.contents['y'].values[df.loc[i]["id"]])

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    id_array = np.array(ids)

    return y_pred, y_true, id_array
