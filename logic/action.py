"""Executes actions for parsed canonical utterances.

This file implements routines to take actions in the conversation, returning
outputs to the user. Actions in the conversation are called `operations` and
are things like running an explanation or performing filtering.
"""
from flask import Flask

from actions.get_action_functions import get_all_action_functions_map, SIMULATION_OPERATIONS
from logic.conversation import Conversation

app = Flask(__name__)


def run_action(conversation: Conversation,
               parse_tree,
               parsed_string: str,
               simulation: bool = False,
               actions=get_all_action_functions_map(),
               build_temp_dataset: bool = True) -> str:
    """Runs the action and updates the conversation object

    Arguments:
        simulation: for simulation test
        build_temp_dataset: Whether to use the temporary dataset stored in the conversation
                            or to rebuild the temporary dataset from scratch.
        actions: The set of avaliable actions
        parsed_string: The grammatical text
        conversation: The conversation object, see `conversation.py`
        parse_tree: The parse tree of the canonical utterance. Note, currently, this is not used,
                    and we compute the actions from the parsed text.
    """
    if parse_tree:
        pretty_parse_tree = parse_tree.pretty()
        app.logger.info(f'Parse tree {pretty_parse_tree}')

    return_statement = ''

    parsed_text = parsed_string.split(' ')

    # Will rebuilt the temporary dataset if requested (i.e, for filtering from scratch)
    # IL: ...or if previousfilter is not part of the parse
    if build_temp_dataset:  # and "previousfilter" not in parsed_text:
        conversation.build_temp_dataset()

    is_or = False

    for i, p_text in enumerate(parsed_text):
        if parsed_text[i] in actions:
            if p_text in SIMULATION_OPERATIONS:
                action_return, action_status = actions[p_text](
                    conversation, parsed_text, i, simulation, is_or=is_or)
            else:
                action_return, action_status = actions[p_text](
                    conversation, parsed_text, i, is_or=is_or)
            return_statement += action_return

            # If operation fails, return error output to user
            if action_status == 0:
                break

            # This is a bit ugly but basically if an or occurs
            # we hold onto the or until we need to flip it off, i.e. filtering
            # happens
            if is_or is True and actions[p_text] == 'filter':
                is_or = False

        if p_text == 'or':
            is_or = True

    # Store 1-turn parsing
    conversation.store_last_parse(parsed_string)

    while return_statement.endswith("<br>"):
        return_statement = return_statement[:-len("<br>")]

    return return_statement
