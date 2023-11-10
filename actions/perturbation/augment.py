"""Data augmentation operation."""

import nlpaug.augmenter.word as naw

from actions.prediction.predict import prediction_generation


def augment_operation(conversation, parse_text, i, **kwargs):
    """Data augmentation."""
    data = conversation.temp_dataset.contents['X']
    idx = None

    if conversation.custom_input is not None and conversation.used is False:
        if conversation.describe.get_dataset_name() == "covid_fact":
            claim, evidence = conversation.custom_input['first_input'], conversation.custom_input['second_input']
        else:
            # TODO
            pass
    else:
        assert len(conversation.temp_dataset.contents["X"]) == 1

        try:
            idx = conversation.temp_dataset.contents["X"].index[0]
        except ValueError:
            return "Sorry, invalid id", 1

        if conversation.describe.get_dataset_name() == "covid_fact":
            claim = conversation.get_var("dataset").contents["X"].iloc[idx]["claims"]
            evidence = conversation.get_var("dataset").contents["X"].iloc[idx]["evidences"]
        else:
            # TODO
            pass

    return_s = ""

    _, pre_prediction = prediction_generation(data, conversation, idx, num_shot=3, given_first_field=None, given_second_field=None)

    # Word augmenter
    if conversation.describe.get_dataset_name() == "covid_fact":
        aug = naw.SynonymAug(aug_src='wordnet')

        # Augment both claim and evidence to create a new instance
        augmented_first_field = aug.augment(claim)
        augmented_second_field = aug.augment(evidence)

        return_s += f"Instance of ID <b>{idx}</b> <br>"
        return_s += f"<b>Claim</b>: {claim}<br>"
        return_s += f"<b>Original evidence:</b> {evidence}<br>"
        return_s += f"<b>Prediction before augmentation</b>: <span style=\"background-color: #6CB4EE\">{pre_prediction}</span><br>"
        return_s += f"<b>Augmented claim:</b> {augmented_first_field}<br>"
        return_s += f"<b>Augmented evidence:</b> {augmented_second_field}<br>"
    else:
        # TODO
        pass
    _, post_prediction = prediction_generation(data, conversation, idx, num_shot=3, given_first_field=augmented_first_field, given_second_field=augmented_second_field)
    return_s += f"<b>Prediction after augmentation</b>: <span style=\"background-color: #6CB4EE\">{post_prediction}</span>"

    return return_s, 1
