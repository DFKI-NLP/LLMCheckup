"""Show data labels"""
import gin
import numpy as np

from actions.util_functions import get_parse_filter_text
from timeout import timeout


@timeout(60)
@gin.configurable
def show_labels_operation(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows labels."""
    y_values = conversation.temp_dataset.contents['y']
    intro_text = get_parse_filter_text(conversation)

    if len(y_values) == 0:
        return "There are no instances in the data that meet this description.", 0
    if len(y_values) == 1:
        label = y_values.item()
        label_text = conversation.get_class_name_from_label(label)
        return_string = f"{intro_text} the label is <b>{label_text}</b>."
    else:
        return_string = f"{intro_text}<br><br>"

        labels = np.array(y_values)
        length = len(labels)
        num_class = len(conversation.class_names)
        class_counter = [0 for i in range(num_class)]

        for i in labels:
            class_counter[i] += 1

        for i in range(len(class_counter)):
            class_counter[i] = round(class_counter[i] / length * 100, conversation.rounding_precision)

        return_string += '<ul>'
        for i in range(num_class):
            return_string += "<li>"
            return_string += f'<b>{class_counter[i]}%</b> of instances have label <span style=\"background-color: ' \
                             f'#6CB4EE\">{conversation.class_names[i]}</span>'
            return_string += "</li>"
        return_string += '</ul>'

    return return_string, 1
