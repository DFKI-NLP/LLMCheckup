import random
import time
from actions.prediction.predict import prediction_generation


def random_prediction(conversation, parse_text, i, **kwargs):
    """randomly pick an instance from the dataset and make the prediction
    `randompredict [E]`
    """
    data = conversation.temp_dataset.contents['X']

    try:
        precomputation_num = int(parse_text[i+1])
    except ValueError:
        precomputation_num = 5

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    random.seed(time.time())
    f_names = list(data.columns)

    random_num_ls = [random.randint(0, len(data[f_names[0]])) for _ in range(precomputation_num)]

    for random_num in random_num_ls:
        _, prediction = prediction_generation(data, conversation, random_num, external_search=False)

    css_style = "style='border: 5px solid #dddddd;text-align: left;padding: 8px;'"

    return_s = "Precomputed predictions as follows:<br>"

    return_s += f"<table><tr><th {css_style}>id</th><th {css_style}>prediction</th></tr>"

    df = conversation.precomputation_of_prediction
    for i in range(len(df)):
        if conversation.describe.get_dataset_name() == "covid_fact":
            prediction = conversation.class_names[df.loc[i]['prediction']]
        else:
            choice = data["choices"].values[df.loc[i]["id"]]
            idx = df.loc[i]['prediction']
            prediction = choice.split("-")[idx]
        return_s += f"<tr><td {css_style}>{df.loc[i]['id']}</td><td {css_style}>{prediction}</td></tr>"

    return_s += "</table>"

    return return_s, 1
