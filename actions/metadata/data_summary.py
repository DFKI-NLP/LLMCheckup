"""Data summary operation."""
from timeout import timeout


@timeout(60)
def data_operation(conversation, parse_text, i, **kwargs):
    """Data summary operation."""
    text = ""

    description = conversation.describe.get_dataset_description()
    text += f"The data contains information related to <b>{description}</b>.<br>"

    # List out the feature names
    f_names = list(conversation.temp_dataset.contents['X'].columns)

    f_string = "<ul>"
    for fn in f_names:
        f_string += f"<li>{fn}</li>"
    f_string += "</ul>"
    text += f"The exact <b>feature names</b> in the data are listed as follows:{f_string}<br>"

    class_list = list(conversation.class_names.values())
    text += "The dataset has following <b>labels</b>: "
    text += "<ul>"
    for i in range(len(class_list)):
        text += "<li>"
        text += str(class_list[i])
        text += "</li>"
    text += "</ul><br>"

    return text, 1
