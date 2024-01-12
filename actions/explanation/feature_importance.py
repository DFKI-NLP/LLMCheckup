import inseq

from actions.prediction.predict import prediction_generation, convert_str_to_options
from inseq.data.aggregator import SequenceAttributionAggregator, SubwordAggregator


SUPPORTED_METHODS = ["integrated_gradients", "attention", "lime", "input_x_gradient"]


def handle_input(parse_text, i):
    """
    Handle the parse text and return the list of numbers(ids) and topk value if given
    Args:
        parse_text: parse_text from bot

    Returns: method, topk

    """
    if parse_text[i + 1] == "all":
        topk = 5
    else:
        try:
            topk = int(parse_text[i + 2])
        except ValueError:
            topk = 5

    method_name = "attention"
    try:
        if parse_text[i + 1] in SUPPORTED_METHODS:
            # Without topk
            method_name = parse_text[i + 1]
        elif parse_text[i + 2] in SUPPORTED_METHODS:
            # with all
            method_name = parse_text[i + 2]
        elif parse_text[i + 3] in SUPPORTED_METHODS:
            # with topk value
            method_name = parse_text[i + 3]
    except IndexError:
        pass
    return topk, method_name


def k_highest_indices(lst, k):
    # Create a list of tuples (value, index)
    indexed_lst = list(enumerate(lst))

    # Sort the list by the values in descending order
    sorted_lst = sorted(indexed_lst, key=lambda x: x[1], reverse=True)

    # Extract the first k indices
    highest_indices = [index for index, value in sorted_lst[:k]]

    return highest_indices


def feature_importance_operation(conversation, parse_text, i, **kwargs) -> (str, int):
    """
    feature attribution operation
    Args:
        conversation
        parse_text: parsed text from T5
        i: counter pointing at operation
        **kwargs:

    Returns:
        formatted string
    """
    # filter id 213 and nlpattribute all [E]
    # filter id 33 and nlpattribute topk 1 [E]

    # Get topk value and feature attribution method name
    topk, method_name = handle_input(parse_text, i)

    model = conversation.decoder.gpt_model

    inseq_model = inseq.load_model(
        model,
        method_name,
        device=str(conversation.decoder.gpt_model.device.type),  # Use same device as already loaded GPT model
    )

    dataset = conversation.temp_dataset.contents["X"]

    if conversation.describe.get_dataset_name() == "covid_fact":
        # COVID-Fact dataset processing
        first_field_name = "claims"
        second_field_name = "evidences"
    else:
        # ECQA
        first_field_name = "texts"
        second_field_name = "choices"

    try:
        idx = dataset.index[0]
        second_field = dataset[second_field_name].item()
        first_field = dataset[first_field_name].item()
    except:
        idx = None
        first_field, second_field = conversation.custom_input['first_input'], conversation.custom_input['second_input']

    # Get model prediction
    _, prediction = prediction_generation(dataset, conversation, idx, external_call=True, external_search=False)

    if conversation.describe.get_dataset_name() == "covid_fact":
        # TODO: Import prompt from some central prompts file which are also used in prediction
        input_text = (f"Your task is to predict the veracity of the claim based on the evidence. \n"
                      f"Evidence: '{second_field}' \n"
                      f"Claim: '{first_field}' \n"
                      f"Please provide your answer as one of the labels: Refuted or Supported. \n"
                      f"Veracity prediction:")
    else:
        input_text = (f"Each 3 items in the following list contains the question, choice and prediction. Your task "
                      f"is to choose one of the choices as the answer for the question.\n"
                      f"Question: '{first_field}'\n"
                      f"Choice: '{convert_str_to_options(second_field)}'\n"
                      f"Prediction:")
        prediction = second_field.split("-")[int(prediction)]

    # Store current system prompt for feature attribution
    conversation.current_prompt = input_text

    # Attribute text
    out = inseq_model.attribute(
        input_texts=input_text,
        generated_texts=f"{input_text} {prediction}",
        attribute_target=False,
        n_steps=1,
        step_scores=["probability"],
        show_progress=True,
        generation_args={}
    )

    # First we aggregate the subword tokens.
    # The second aggregate call is exactly like the one above:
    # for attention, [mean, mean] (mean across the layers and heads dimensions)
    out_agg = out.aggregate(SubwordAggregator).aggregate()

    len_generation = out_agg[0].target_attributions.shape[1]
    # Extract 1D heatmap (attributions for final token)
    final_token_attributions = out_agg[0].target_attributions[:, len_generation-1]
    topk_tokens = [out_agg[0].target[i].token for i in k_highest_indices(final_token_attributions, topk)]

    # TODO: Possibly reduce to tokens in "claim" and "evidence" (exclude the prompt)

    # Get HTML visualization from Inseq
    heatmap_viz = out_agg.show(return_html=True, do_aggregation=False).split("<html>")[1].split("</html>")[0]

    # TODO: Find sensible verbalization

    return_s = f"<b>Feature attribution method: </b>{method_name}<br>"
    return_s += f"Top {topk} token(s):<br>"

    for i in topk_tokens:
        if i == "<s>":  # This token causes strikethrough text in HTML! 🤨
            i = "< s >"
        return_s += f"<b>{i}</b><br>"

    return_s += "<details><summary>"
    return_s += "The visualization: "
    return_s += "</summary>"
    return_s += heatmap_viz
    return_s += "</details><br>"

    return return_s, 1
