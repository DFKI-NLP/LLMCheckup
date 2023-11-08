"""Last turn operation."""
from copy import deepcopy

from actions.about.function import function_operation
from actions.about.self import self_operation
from actions.context.followup import followup_operation
from actions.metadata.data_summary import data_operation
from actions.metadata.define import define_operation
from actions.metadata.labels import show_labels_operation
from actions.metadata.model import model_operation
from actions.metadata.show_data import show_operation
from actions.prediction.mistakes import show_mistakes_operation
from actions.prediction.predict import predict_operation
from actions.prediction.score import score_operation


def get_most_recent_ops(operations, op_list):
    for op in operations:
        for word in op.split(' '):
            if word in op_list:
                return op
    return None


def last_turn_operation(conversation, parse_text, i, **kwargs):
    """Last turn operation.

    The function computes the last set of operations (excluding filtering) on the
    current temp_dataset. This feature enables things like doing filtering and then
    running whatever set of operations were run last.
    """

    # Just get the operations run in the last parse
    last_turn_operations = conversation.get_last_parse()[::-1]

    # Store the conversation
    last_turn_conversation = deepcopy(conversation)

    # Remove the filter operations so that only the actions
    # like explantions or predictions will be run
    # TODO(dylan): find a way to make this a bit cleaner... right now both this function
    # and the dictionary in get_action_function need to be updated with new functions. We can't
    # import from that file because of circular imports... should be a way to do this better though
    # so you don't have to update in both places.
    excluding_filter_ops = {
        #'logic': explain_operation,
        'predict': predict_operation,
        'self': self_operation,
        'previousoperation': last_turn_operation,
        'data': data_operation,
        'followup': followup_operation,
        #'important': important_operation,
        'show': show_operation,
        'model': model_operation,
        'function': function_operation,
        'score': score_operation,
        'label': show_labels_operation,
        'mistake': show_mistakes_operation,
        #'change': what_if_operation,
        'define': define_operation
    }

    most_recent_ops = get_most_recent_ops(last_turn_operations, excluding_filter_ops)

    # in case we can't find anything
    if most_recent_ops is None:
        return "", 1

    # Delete most recent op from prev parses so that we don't recurse to it again
    # if we don't do this, it may recurse indefinetly
    for i in range(len(last_turn_conversation.last_parse_string)):
        j = len(last_turn_conversation.last_parse_string) - i - 1
        if last_turn_conversation.last_parse_string[j] == most_recent_ops:
            del last_turn_conversation.last_parse_string[j]

    parsed_text = most_recent_ops.split(' ')
    return_statement = ''
    for i, p_text in enumerate(parsed_text):
        if parsed_text[i] in excluding_filter_ops:
            action_output, action_status = excluding_filter_ops[p_text](last_turn_conversation,
                                                                        parsed_text, i,
                                                                        is_or=False)
            return_statement += action_output
            if action_status == 0:
                break

    return return_statement, 1
