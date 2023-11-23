import pandas as pd


def get_metadata_by_model_name(model_name: str) -> str:
    """
    Get metadata about the deployed models
    :param model_name: model name
    :return: metadata information
    """
    df = pd.read_csv("./cache/metadata.csv")
    idx = df.index[df['model_name'] == model_name].tolist()

    return_s = f"<b>Model name:</b> {model_name}, <b>Model size: </b> {df['size'].loc[idx].values[0]}<br>"
    return_s += f"<b>Model description: </b> {df['description'].loc[idx].values[0]}<br>"
    return_s += f"<b>Training data: </b> {df['training_data'].loc[idx].values[0]}<br>"

    return return_s
