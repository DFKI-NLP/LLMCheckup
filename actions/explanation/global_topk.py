from actions.explanation.topk import topk
from timeout import timeout


@timeout(60)
def global_top_k(conversation, parse_text, i, **kwargs):

    # Set default
    k = 10

    if "all" in parse_text:
        k = 10
    else:
        for item in parse_text:
            try:
                k = int(item)
            except:
                pass

    class_names = conversation.class_names

    first_argument = parse_text[i + 1]
    class_idx = None

    if first_argument in list(class_names.values()):
        class_idx = first_argument

    reverse = True

    if "least" in parse_text:
        reverse = False

    return topk(conversation, k, class_idx=class_idx, reverse=reverse), 1
