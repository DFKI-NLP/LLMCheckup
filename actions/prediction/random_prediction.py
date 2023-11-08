import random
import time
from actions.prediction.predict import prediction_generation
from timeout import timeout


@timeout(60)
def random_prediction(conversation, parse_text, i, **kwargs):
    """randomly pick an instance from the dataset and make the prediction
    `randompredict [E]`
    """
    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    # return_s = ""

    random.seed(time.time())
    f_names = list(data.columns)

    random_num = random.randint(0, len(data[f_names[0]]))
    # filtered_text = ''

    # for f in f_names[:1]:
    #     filtered_text += data[f][random_num]
    #
    # return_s += f"The random text is with <b>id {random_num}</b>: <br><br>"
    # return_s += "<ul>"
    # return_s += "<li>"
    # return_s += f'The text is: {filtered_text}'
    # return_s += "</li>"
    #
    # return_s += "<li>"

    return_s, _ = prediction_generation(data, conversation, random_num)

    return return_s, 1
