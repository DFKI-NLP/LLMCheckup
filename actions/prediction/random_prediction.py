import random
import time
from actions.prediction.predict import handle_input
from timeout import timeout


@timeout(60)
def random_prediction(conversation, parse_text, i, **kwargs):
    """randomly pick an instance from the dataset and make the prediction
    `randompredict [E]`
    """
    model = conversation.get_var('model').contents
    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    text = handle_input(parse_text)

    return_s = ''

    random.seed(time.time())
    f_names = list(data.columns)

    # Using random.randint doesn't work here somehow
    random_num = random.randint(0, len(data[f_names[0]]))
    filtered_text = ''

    for f in f_names[:1]:
        filtered_text += data[f][random_num]

    return_s += f"The random text is with <b>id {random_num}</b>: <br><br>"
    return_s += "<ul>"
    return_s += "<li>"
    return_s += f'The text is: {filtered_text}'
    return_s += "</li>"

    return_s += "<li>"
    model_predictions = model.predict(data, text, conversation)
    if conversation.class_names is None:
        prediction_class = str(model_predictions[random_num])
        return_s += f"The class name is not given, the prediction class is <b>{prediction_class}</b>"
    else:
        try:
            class_text = conversation.class_names[model_predictions[random_num]]
        except KeyError:
            class_text = model_predictions[random_num]
        return_s += f"The prediction is <span style=\"background-color: #6CB4EE\">{class_text}</span>."
    return_s += "</li>"
    return_s += "</ul>"

    return return_s, 1
