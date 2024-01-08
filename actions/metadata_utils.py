import pandas as pd

MODEL_NAME = ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-v0.1,7B"]


def get_metadata_by_model_name(model_name: str) -> str:
    """
    Get metadata about the deployed models
    :param model_name: model name
    :return: metadata information
    """
    if "Llama" in model_name:
        model_name = MODEL_NAME[0]
        gptq = True

    if "Mistral" in model_name:
        model_name = MODEL_NAME[1]
        gptq = True

    df = pd.read_csv("./cache/metadata.csv")
    idx = df.index[df['model_name'] == model_name].tolist()

    return_s = f"<b>Model name:</b> {model_name}, <b>Model size: </b> {df['size'].loc[idx].values[0]}<br>"

    if gptq:
        return_s += "The model is quantized by GPTQ. "
    return_s += f"<b>Model description: </b> {df['description'].loc[idx].values[0]}<br>"
    return_s += f"<b>Training data: </b> {df['training_data'].loc[idx].values[0]}<br>"

    return return_s
