"""Data augmentation operation."""

import nlpaug.augmenter.word as naw


def augment_operation(conversation, parse_text, i, **kwargs):
    """Data augmentation."""
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

    # Word augmenter
    if conversation.describe.get_dataset_name() == "covid_fact":
        aug = naw.SynonymAug(aug_src='wordnet')
        augmented_evidence = aug.augment(evidence)
        return_s += f"Instance of ID <b>{idx}</b> <br>"
        return_s += f"<b>Claim</b>: {claim}<br>"
        return_s += f"<b>Original evidence:</b> {evidence}<br>"
        return_s += f"<b>Augmented evidence:</b> {augmented_evidence}<br>"

    else:
        # TODO
        pass

    return return_s, 1
