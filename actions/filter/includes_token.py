import string
import re

from timeout import timeout


@timeout(60)
def includes_operation(conversation, parse_text, i, **kwargs):
    text_to_match = conversation.include_word
    # if "includetoken" in text_to_match:
    #     return "Sorry, I could not find any match!", 1
    # remove the quotes around the text
    while len(text_to_match) > 0 and text_to_match[0] in string.punctuation:
        text_to_match = text_to_match[1:]
    while len(text_to_match) > 0 and text_to_match[-1] in string.punctuation:
        text_to_match = text_to_match[:-1]
    text_to_match = re.escape(text_to_match)
    temp_dataset = conversation.temp_dataset.contents["X"]

    dataset_name = conversation.describe.get_dataset_name()

    # Filter temp dataset for subsequent actions
    filtered_df = temp_dataset[temp_dataset[conversation.text_fields].apply(
        lambda row: row.str.contains(text_to_match)).any(axis=1)]
    conversation.temp_dataset.contents["X"] = filtered_df

    if dataset_name == 'boolq':
        questions = temp_dataset["question"]
        passages = temp_dataset["passage"]
        text_inputs = []
        for i in range(len(questions)):
            text_inputs.append(questions[i] + " " + passages[i])
    elif dataset_name == 'daily_dialog':
        text_inputs = temp_dataset["dialog"]
    else:
        text_inputs = temp_dataset["text"]

    threshold = 3  # num of words before and after the match
    max_num_of_matches = 7  # max num of words that we output as matches
    output_str = ""

    # Return with empty string if filter result was already shown in a previous turn
    last_parses = conversation.last_parse_string
    if "previousfilter" in parse_text:
        return output_str, 1
    if len(last_parses) >= 2:
        if "includes and" in last_parses[-1]:
            return output_str, 1

    match_len = len(text_to_match)
    total_matches = 0
    for inum, text_input in enumerate(text_inputs):
        if total_matches >= max_num_of_matches:
            break
        matched_indices = [m.start() for m in re.finditer(text_to_match, text_input, re.IGNORECASE)][
                          :max_num_of_matches]
        total_matches += len(matched_indices)
        for match_idx in matched_indices:
            before = text_input[:match_idx]
            match = text_to_match
            after = text_input[match_idx + match_len:]
            before_short = ' '.join(before.split()[-threshold:])
            after_short = ' '.join(after.split()[:threshold])
            idx = temp_dataset.index[inum]
            #print(temp_dataset.index)
            output_str += f"idx {idx}: "
            if len(before_short) > 0:
                output_str += f"<details><summary>... {before_short} "
            output_str += f"<b>{match}</b>"
            if len(after_short) > 0:
                output_str += f" {after_short} ...</summary>Text: {before}<b>{match}</b>{after}</details>"
            output_str += " <br>"
    if len(output_str) == 0:
        output_str = "No matches were found for: " + text_to_match + "<br>"
    else:
        output_str = f"I found the following matches for <b>{text_to_match}</b>: <br>" + output_str

    return output_str, 1
