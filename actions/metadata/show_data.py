"""Function to show data instances.

For single instances, this function prints out the feature values. For many instances,
it returns the mean.
"""
import gin

from actions.util_functions import get_parse_filter_text
from typing import List

from timeout import timeout


def summarize_consecutive_ids(instance_ids: List[int]):
    if not instance_ids:
        return ""

        # Sort the list
    instance_ids = sorted(instance_ids)

    # Initialize variables
    summary = []
    start = instance_ids[0]
    end = instance_ids[0]

    # Iterate through the list
    for i in range(1, len(instance_ids)):
        if instance_ids[i] == end + 1:
            end = instance_ids[i]
        else:
            if start == end:
                summary.append(str(start))
            else:
                summary.append(f"{start}-{end}")

            start = end = instance_ids[i]

    # Add the last range or single integer
    if start == end:
        summary.append(str(start))
    else:
        summary.append(f"{start}-{end}")

    # Return the summarized string
    return ", ".join(summary)


@timeout(60)
@gin.configurable
def show_operation(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows an instance."""
    data = conversation.temp_dataset.contents["X"]

    intro_text = get_parse_filter_text(conversation)
    rest_of_info_string = "The rest of the features are<br><br>"
    init_len = len(rest_of_info_string)
    if len(data) == 0:
        return "There are no instances in the data that meet this description.", 0
    if len(data) == 1:
        return_string = f"{intro_text} the features are<br><br>"

        for i, feature_name in enumerate(data.columns):
            feature_value = data[feature_name].values[0]
            text = f"<b>{feature_name}</b>: {feature_value}<br>"
            if i < n_features_to_show:
                return_string += text
            else:
                rest_of_info_string += text
    else:
        instance_ids = list(data.index)
        str_instance_ids = summarize_consecutive_ids(instance_ids)
        return_string = f"{intro_text} the instance id's are:<br><br>"
        return_string += str_instance_ids
        return_string += "<br><br>Which one do you want to see?<br><br>"

    # If we've written additional info to this string
    if len(rest_of_info_string) > init_len:
        return_string += "<br><br>I've truncated this instance to be concise. Let me know if you"
        return_string += " want to see the rest of it.<br><br>"
        conversation.store_followup_desc(rest_of_info_string)
    return return_string, 1
