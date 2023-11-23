"""Describes the model."""
from actions.metadata_utils import get_metadata_by_model_name
from timeout import timeout


@timeout(60)
def model_operation(conversation, parse_text, i, **kwargs):
    """Model description."""
    objective = conversation.describe.get_dataset_objective()
    model = conversation.describe.get_model_description()
    text = f"I use a <em>{model}</em> model to {objective}.<br><br>"

    model_name = conversation.decoder.parser_name

    text += get_metadata_by_model_name(model_name)

    return text, 1
