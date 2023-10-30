import nltk

from timeout import timeout


flag2bool = {"least": False, "most": True}


def get_frequent_words(conversation, f_names, top=5, reverse=True):
    """

    Args:
        conversation: conversation object
        f_names: list of feature names
        top: top k frequent words

    Returns:
        frequent_words: list of tuples in form: (word, freq)
    """
    df = conversation.temp_dataset.contents["X"]

    nltk.download("stopwords")
    sw = nltk.corpus.stopwords.words("english")
    temp = ""
    for f in f_names:
        for inum, t in enumerate(df[f]):
            temp += str(t) + " "

    words = temp.split(" ")

    words_ne = []
    for word in words:
        if word not in sw:
            words_ne.append(word)

    word_dict = dict(nltk.FreqDist(words_ne))

    frequent_words = sorted(word_dict.items(), key=lambda x: x[1], reverse=reverse)[:top]

    text = "<table style='border: 1px solid black;'>"
    text += "<tr style='border: 1px solid black;'>"
    text += "<th> Word </th>"
    text += "<th> Frequence </th>"
    text += "</tr>"

    for i in range(len(frequent_words)):
        text += "<tr style='border: 1px solid black;'>"
        text += f"<td style='border: 1px solid black;'> {frequent_words[i][0]} </td>"
        text += f"<td style='border: 1px solid black;'> {frequent_words[i][1]} </td>"
        text += "</tr>"
    text += "</table><br>"

    return text


@timeout(60)
def keyword_operation(conversation, parse_text, i, **kwargs):
    """topk keywords operation. """
    df = conversation.temp_dataset.contents["X"]

    # List out the feature names
    f_names = list(df.columns)

    reverse = True

    # Extract topk value
    if "keywords all" in " ".join(parse_text):
        top = 25
        if parse_text[i+2] in flag2bool.keys():
            reverse = flag2bool[parse_text[i+2]]
    else:
        top = None
        try:
            if int(parse_text[i+1]):
                top = int(parse_text[i+1])
        except ValueError:
            pass

        if top is None:
            top = 25

            if parse_text[i+1] in flag2bool.keys():
                reverse = flag2bool[parse_text[i+1]]
        else:
            if parse_text[i+2] in flag2bool.keys():
                reverse = flag2bool[parse_text[i+2]]

    if reverse:
        return_s = f"The {top} most frequent word(s) in the dataset: <br>"
    else:
        return_s = f"The {top} least frequent word(s) in the dataset: <br>"
    return_s += get_frequent_words(conversation, f_names=f_names, top=top, reverse=reverse)

    return return_s, 1
