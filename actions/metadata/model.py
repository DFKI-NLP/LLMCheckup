"""Describes the model."""

from timeout import timeout


@timeout(60)
def model_operation(conversation, parse_text, i, **kwargs):
    """Model description."""
    objective = conversation.describe.get_dataset_objective()
    model = conversation.describe.get_model_description()
    text = f"I use a <em>{model}</em> model to {objective}. I was trained on Pile dataset as an autoregressive " \
           f"language model, using cross-entropy loss to maximize the likelihood of predicting the next token " \
           f"correctly.<br><br>"

    return text, 1
