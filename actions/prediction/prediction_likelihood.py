import json

import numpy as np
import torch
from torch import nn

from timeout import timeout

SINGLE_INSTANCE_TEMPLATE = """
The model predicts the instance with <b>{filter_string}</b> as:
<b>
"""


def extract_id_from_parse_text(parse_text):
    """

    Args:
        parse_text: parsed text from T5

    Returns:
        id of instance
    """
    instance_id = None
    for item in parse_text:
        try:
            if int(item):
                instance_id = int(item)
        except:
            pass
    return instance_id


def get_predictions_and_probabilities(name, instance_id, dataset_name, conversation):
    """

    Args:
        name: dataset name
        instance_id: id of instance

    Returns:
        predictions and probabilities
    """
    if dataset_name != 'daily_dialog':
        data_path = f"./cache/{name}/ig_explainer_{name}_explanation.json"
    else:
        data_path = f"./cache/{name}/ig_explainer_{name}_prediction.json"
    fileObject = open(data_path, "r")
    jsonContent = fileObject.read()
    json_list = json.loads(jsonContent)

    prediction = json_list[instance_id]["predictions"]

    model_predictions = np.argmax(prediction)
    if dataset_name == "boolq":
        model_prediction_probabilities = (nn.Softmax(dim=0)(torch.tensor(prediction))).detach().numpy()
    elif dataset_name == "daily_dialog":
        model_prediction_probabilities = (nn.Softmax(dim=0)(torch.Tensor(prediction).float())).detach().numpy()
    elif dataset_name == 'olid':
        model_prediction_probabilities = (nn.Softmax(dim=0)(torch.Tensor(prediction).float())).detach().numpy()
    else:
        raise NotImplementedError(f"{dataset_name} is not supported!")

    return model_predictions, model_prediction_probabilities


@timeout(60)
def predict_likelihood(conversation, parse_text, i, **kwargs):
    """The prediction likelihood operation."""
    # `filter id 15 and likelihood [E]`

    # Get the dataset name
    name = conversation.describe.get_dataset_name()
    instance_id = extract_id_from_parse_text(parse_text)

    dataset_name = conversation.describe.get_dataset_name()

    if instance_id is not None:
        model_predictions, model_prediction_probabilities = get_predictions_and_probabilities(name, instance_id, dataset_name, conversation)
    else:
        # TODO: likelihood for a certain class
        raise ValueError("ID is not given")

    return_s = f"For instance with id <b>{instance_id}</b>: "
    return_s += "<ul>"

    # Go through all classes
    for _class in range(len(model_prediction_probabilities)):
        class_name = conversation.get_class_name_from_label(_class)
        prob = model_prediction_probabilities
        prob = round(model_prediction_probabilities[_class] * 100, conversation.rounding_precision)
        return_s += "<li>"
        return_s += f"The likelihood of class <span style=\"background-color: #6CB4EE\">{class_name}</span> is <b>{prob}%</b>"
        return_s += "</li>"
    return_s += "</ul>"

    return return_s, 1

