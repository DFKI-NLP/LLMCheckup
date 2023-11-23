from sentence_transformers import SentenceTransformer, util


def extract_id_number(parse_text):
    """
    Args:
        parse_text: parsed text from conversation
    Returns:
        id of text and number of cfe instances
    """
    num_list = []
    for item in parse_text:
        try:
            if int(item):
                num_list.append(int(item))
        except:
            pass
    if len(num_list) == 1:
        return num_list[0], 1
    elif len(num_list) == 2:
        return num_list[0], num_list[1]
    else:
        raise ValueError("Too many numbers in parse text!")


def similar_instances_operation(conversation, parsed_text, i, **kwargs):
    """
    Args:
        conversation: conversation object
        parsed_text: parsed text from conversation
        i: index of operation
    Returns:
        final_results  matched results
    """
    if len(conversation.temp_dataset.contents["X"]) == 0:
        return "There are no instances that meet this description!", 0
    dataset = conversation.stored_vars["dataset"]

    if conversation.describe.get_dataset_name() == "covid_fact":
        feature_name = "claims"
    else:
        feature_name = "texts"

    if conversation.custom_input is not None and conversation.used is False:
        query = conversation.custom_input["first_field"]
        idx = None

        try:
            number = int(parsed_text[i+1])
        except ValueError:
            number = 3

    else:
        idx, number = extract_id_number(parsed_text)
        data = conversation.temp_dataset.contents['X']
        if conversation.describe.get_dataset_name() == "covid_fact":
            query = data[feature_name].values[0]
        else:
            query = data[feature_name].values[0]

    final_results = get_similar_str(query, idx, number, dataset, conversation, feature_name)

    return final_results, 1


def get_similar_str(query, idx, number, dataset, conversation, feature_name):
    """
    Args:
    dataset: dataset from the conversation
    query_sentence: query sentence to be matched
    query_label: label to search for in the dataset
    number: number of hits to be returned
    Returns:
    filtered similarity response to a maximum of specified number
    """
    # preparing the output string
    if idx is not None:
        out_str = "The original text for <b>id " + str(idx) + "</b>:<br>"
    else:
        out_str = f"Similarity of custom input <i>{query}</i>:<br>"

    query_tokens = query.split()
    query_preview = " ".join(query_tokens[:16])
    out_str += "<summary>" + query_preview + "...</summary><details>" + query + "</details><br>"
    out_str += "Here are some instances similar to <b>id " + str(idx) + "</b>:<br>"
    found_similars = get_similars(query, idx, dataset, number, conversation, feature_name)
    for cossim, similar_id, similar in found_similars:
        similar_tokens = similar.split()
        similar_preview = " ".join(similar_tokens[:16])
        out_str += "<b> id " + str(similar_id) + "</b> (cossim " + str(round(cossim,
                                                                             3)) + "):  <summary>" + similar_preview + "...</summary><details>" + similar + "</details><br>"
    return out_str


def get_similars(query, query_idx, dataset, number, conversation, feature_name):
    """
    Args:
    dataset: dataset from the conversation
    query_sentence: query sentence to be matched
    query_label: label to search for in the dataset
    Returns:
    filtered similarity response
    """

    # computing similarities
    indices = []
    texts = []
    for idx in list(dataset.contents["X"].index):
        if idx != query_idx:
            indices.append(idx)
            texts.append(dataset.contents["X"].loc[[idx]][feature_name].values[0])

    device = str(conversation.decoder.gpt_model.device.type)

    similarity_model = conversation.prompts.sentence_emb_model.to(device)
    query_embedding = similarity_model.encode(query, convert_to_tensor=True).to(device)
    sent_embeddings = similarity_model.encode(texts, convert_to_tensor=True).to(device)
    cos_sim = util.cos_sim(query_embedding, sent_embeddings)[0].tolist()

    # sort by cossim
    similars = []
    for i in range(len(cos_sim)):
        similars.append((cos_sim[i], indices[i], texts[i]))
    similars = sorted(similars, key=lambda x: x[0], reverse=True)
    return similars[:number]
