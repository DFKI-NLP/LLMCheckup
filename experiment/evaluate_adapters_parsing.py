import numpy as np
import os
import re
import sys
import torch

from word2number import w2n
import string

from transformers import AutoAdapterModel, AutoTokenizer
from transformers import TextClassificationPipeline, TokenClassificationPipeline
from sentence_transformers import SentenceTransformer, util

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AdapterParse:
    """Evaluating Adapter's Parsing Accuracy."""

    def __init__(self):

        bert_model = "bert-base-uncased"
        self.intent_adapter_model = AutoAdapterModel.from_pretrained(bert_model)
        self.slot_adapter_model = AutoAdapterModel.from_pretrained(bert_model)
        self.adapter_tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.device = 0 if torch.cuda.is_available() else -1

        intent_adapter_path = "./intents_and_slots/intent_slot_classification/adapters/all"
        intent_adapter = self.intent_adapter_model.load_adapter(intent_adapter_path)
        self.intent_adapter_model.load_head(intent_adapter_path)
        self.intent_adapter_model.set_active_adapters([intent_adapter])
        self.slot_adapter_model = AutoAdapterModel.from_pretrained(bert_model)

        slot_adapter_path = "./intents_and_slots/intent_slot_classification/adapters/slots"
        slot_adapter = self.slot_adapter_model.load_adapter(slot_adapter_path)
        self.slot_adapter_model.load_head(slot_adapter_path)
        self.slot_adapter_model.set_active_adapters([slot_adapter])

        self.intent_classifier = TextClassificationPipeline(model=self.intent_adapter_model,
                                                            tokenizer=self.adapter_tokenizer,
                                                            return_all_scores=True, task="all", device=self.device)

        self.slot_tagger = TokenClassificationPipeline(model=self.slot_adapter_model,
                                                       tokenizer=self.adapter_tokenizer, task="slots",
                                                       device=self.device)

        self.quote_pattern = r'(\"|\')[^\"\']*(\"|\')'

        self.all_intents = ["adversarial", "augment", "includes", "cfe", "similar", "predict", "self", "data", "show",
                            "likelihood", "model", "function", "score", "countdata", "label", "mistake count",
                            "mistake sample", "keywords", "nlpattribute", "rationalize", "important", "statistic",
                            "randompredict"]
        self.id2label_str = dict()
        for i, intent_name in enumerate(self.all_intents):
            self.id2label_str[i] = intent_name
        # Mapping slots to the fixed values for scores (as defined in grammar.py)
        self.score_mapper = {"positive predictive value": "ppv", "negative predictive value": "npv", "acc": "accuracy",
                             "prec": "precision", "rec": "recall", "f 1": "f1"}
        self.scores = ["accuracy", "f1", "roc", "precision", "recall", "sensitivity", "specificity", " ppv", "npv"]
        self.score_settings = ["micro", "macro", "weighted"]
        self.intent_with_topk_prefix = ["nlpattribute", "important", "similarity"]

        self.core_slots = {"adversarial": ["id"],
                           "augment": ["id"],
                           "includes": [],
                           "cfe": ["id"],
                           "similar": ["id"],
                           "predict": [],
                           "self": [],
                           "data": [],
                           "show": ["id"],
                           "likelihood": ["id"],
                           "model": [],
                           "function": [],
                           "score": [],
                           "countdata": [],
                           "label": [],
                           "mistake count": [],
                           "mistake sample": [],
                           "keywords": [],
                           "nlpattribute": [],
                           "rationalize": ["id"],
                           "important": [],
                           "statistic": [],
                           "randompredict": []}

        self.intent2slot_pattern = {"adversarial": ["id"],
                                    "augment": ["id"],
                                    "includes": ["includetoken"],
                                    "cfe": ["id", "number"],
                                    "similar": ["id", "number"],
                                    "predict": ["id"],
                                    "self": [],
                                    "data": [],
                                    "show": ["id", "includetoken"],
                                    "likelihood": ["class_names", "includetoken", "id"],
                                    "model": [],
                                    "function": [],
                                    "score": ["includetoken", "metric", "class_names", "data_type"],
                                    "countdata": ["include_token"],
                                    "label": ["includetoken"],
                                    "mistake count": ["includetoken"],
                                    "mistake sample": ["includetoken"],
                                    "keywords": ["number"],
                                    "nlpattribute": ["id", "number", "class_names", "sent_level"],
                                    "rationalize": ["id"],
                                    "important": ["class_names", "include_token", "number"],
                                    "statistic": ["includetoken"],
                                    "randompredict": []}

        self.op2clarification = {"adversarial": "generate adversarial examples",
                                 "augment": "do some data augmentation",
                                 "includes": "filter the dataset by the specified word",
                                 "cfe": "generate counterfactuals",
                                 "similar": "search for similar instances in the dataset",
                                 "self": "describe my capabilities",
                                 "predict": "check the prediction",
                                 "data": "explain the dataset",
                                 "show": "show you the instance from the dataset",
                                 "likelihood": "check the likelihood of the prediction",
                                 "model": "talk about the underlying model",
                                 "function": "talk about the possible actions",
                                 "score": "explain the perfromance in terms of different scores",
                                 "countdata": "count the data points",
                                 "label": "show the labels",
                                 "mistake count": "show how many mistakes the model makes",
                                 "mistake sample": "show which mistakes the model makes",
                                 "keywords": "display some keywords relevant for the dataset",
                                 "nlpattribute": "show you the most important features/tokens",
                                 "rationalize": "provide a rationalization, explain the behaviour of the model",
                                 "important": "show the top token attributions based on the global dataset statistics",
                                 "statistic": "show you some statistics for the dataset",
                                 "randompredict": "show a prediction on a random instance"}

        self.deictic_words = ["this", "that", "it", "here"]

        self.model_slots = ["lr", "epochs", "loss", "optimizer", "task", "model_name", "model_summary"]
        self.model_slot_words_map = {"lr": ["lr", "learning rate"], "epochs": ["epoch"], "loss": ["loss"], "optimizer":["optimizer"], "task":["task", "function"], "model_name": ["name", "call"], "model_summary":["summary", "overview"]}

        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        confirm = ["Yes", "Of course", "I agree", "Correct", "Yeah", "Right", "That's what I meant", "Indeed",
                   "Exactly", "True"]
        disconfirm = ["No", "Nope", "Sorry, no", "I think there is some misunderstanding", "Not right", "Incorrect",
                      "Wrong", "Disagree"]

        data_name = ["inform me test data name", "name of training data", "how is the test set called?",
                     "what's the name of the data?"]
        data_source = ["where does training data come from", "where do you get test data", "the source of the dataset?"]
        data_language = ["show me the language of training data", "language of training data",
                         "tell me the language of testing data", "what's the language of the model?"]
        data_number = ["how many training data is used", "count test data points", "tell me the number of data points"]

        # Compute embedding for data flags
        self.data_name = self.st_model.encode(data_name, convert_to_tensor=True)
        self.data_source = self.st_model.encode(data_source, convert_to_tensor=True)
        self.data_language = self.st_model.encode(data_language, convert_to_tensor=True)
        self.data_number = self.st_model.encode(data_number, convert_to_tensor=True)

        # Compute embeddings for confirm/disconfirm
        self.confirm = self.st_model.encode(confirm, convert_to_tensor=True)
        self.disconfirm = self.st_model.encode(disconfirm, convert_to_tensor=True)

    def get_data_type(self, text: str):
        """Checks the data type (train/test supported)"""
        if "test" in text:
            return "test"
        else:
            return "train"

    def get_data_flag(self, text: str):
        """Checks whether the user asks about specific details of the data"""
        # Compute cosine-similarities
        text = self.st_model.encode(text, convert_to_tensor=True)
        dname_scores = util.cos_sim(text, self.data_name)
        dname_score = torch.mean(dname_scores, dim=-1).item()

        dsource_scores = util.cos_sim(text, self.data_source)
        dsource_score = torch.mean(dsource_scores, dim=-1).item()

        dlang_scores = util.cos_sim(text, self.data_language)
        dlang_score = torch.mean(dlang_scores, dim=-1).item()

        dnum_scores = util.cos_sim(text, self.data_number)
        dnum_score = torch.mean(dnum_scores, dim=-1).item()

        max_score_name = None
        max_score = 0

        for score in [("name", dname_score), ("source", dsource_score), ("language", dlang_score), ("number", dnum_score)]:
            if score[1] > max_score and score[1] > 0.5:
                max_score = score[1]
                max_score_name = score[0]
        return max_score_name

    def has_deictic(self, text):
        for deictic in self.deictic_words:
            if " " + deictic in text.lower() or deictic + " " in text.lower():
                return True
        return False

    def get_intent_annotations(self, intext):
        """Returns intent annotations for user input (using adapters)"""
        text_anno = self.intent_classifier(intext)[0]
        labels = []
        for entry in text_anno:
            labels.append((self.id2label_str[int(entry["label"].replace("LABEL_", ""))], entry["score"]))
        labels.sort(key=lambda x: x[1], reverse=True)
        return labels[:5]

    def get_slot_annotations(self, intext):
        """Returns slot annotations for user input (using adapters)"""
        text_anno = self.slot_tagger(intext)
        intext_chars = list(intext)
        # slot_types = ["class_names", "data_type", "id", "includetoken", "metric", "number", "sent_level"]
        slot2spans = dict()
        for anno in text_anno:
            slot_type = anno["entity"][2:]
            if not (slot_type) in slot2spans:
                slot2spans[slot_type] = []
            slot2spans[slot_type].append((anno["word"], anno["start"], anno["end"], anno["entity"]))
        final_slot2spans = dict()
        for slot_type in slot2spans:
            final_slot2spans[slot_type] = []
            span_starts = [s for s in slot2spans[slot_type] if s[-1].startswith("B-")]
            span_starts.sort(key=lambda x: x[1])
            span_ends = [s for s in slot2spans[slot_type] if s[-1].startswith("I-")]
            span_ends.sort(key=lambda x: x[1])
            for i, span_start in enumerate(span_starts):
                if i < len(span_starts) - 1:
                    next_span_start = span_starts[i + 1]
                else:
                    next_span_start = None
                selected_ends = [s[2] for s in span_ends if
                                 s[1] >= span_start[1] and (next_span_start is None or s[1] < next_span_start[1])]
                if len(selected_ends) > 0:
                    span_end = max(selected_ends)
                else:
                    span_end = span_start[2]
                span_start = span_start[1]
                final_slot2spans[slot_type].append("".join(intext_chars[span_start:span_end]))

        return final_slot2spans

    def clean_up(self, text: str):
        while len(text) > 0 and text[-1] in string.punctuation:
            text = text[:-1]
        return text

    def clean_up_number(self, text: str):
        text = self.clean_up(text)
        try:
            text = w2n.word_to_num(text)
            text = str(text)
        except:
            text = ""
            print(f"value is not a number: {text}")
        return text

    def check_heuristics(self, decoded_text: str, orig_text: str):
        """Checks heuristics for those intents/actions that were identified but their core slots are missing.
        """
        id_adhoc = ""
        number_adhoc = ""
        token_adhoc = ""
        if "includes" in decoded_text:
            indicators = ["word ", "words ", "token ", "tokens "]
            for indicator in indicators:
                if indicator in orig_text:
                    word_start = orig_text.index(indicator) + len(indicator)
                    if word_start < len(orig_text):
                        includeword = orig_text[word_start:]
                        token_adhoc = self.clean_up(includeword)
                        break
            # check for quotes
            in_quote = re.search(self.quote_pattern, orig_text)
            if in_quote is not None:
                token_adhoc = self.clean_up(in_quote.group())
        if "id " in orig_text:
            splitted = orig_text[orig_text.index("id ") + 2:].strip().split()
            if len(splitted) > 0:
                id_adhoc = self.clean_up(splitted[0])
        splitted_text = orig_text.split()
        for tkn in splitted_text:
            if tkn.isdigit() and not (tkn == id_adhoc):
                number_adhoc = tkn
                break
        return id_adhoc, number_adhoc, token_adhoc

    def get_num_value(self, text: str):
        """Converts text to number if possible"""
        for ch in string.punctuation:
            if ch in text:
                text = text.replace(ch, "")
        if len(text) > 0 and not (text.isdigit()):
            try:
                converted_num = w2n.word_to_num(text)
            except:
                converted_num = None
            if converted_num is not None:
                text = str(converted_num)
        if not (text.isdigit()):
            text = ""
        return text

    def is_confirmed(self, text: str):
        """Checks whether the user provides a confirmation or not"""
        # Compute cosine-similarities
        text = self.st_model.encode(text, convert_to_tensor=True)
        confirm_scores = util.cos_sim(text, self.confirm)
        disconfirm_scores = util.cos_sim(text, self.disconfirm)
        confirm_score = torch.mean(confirm_scores, dim=-1).item()
        disconfirm_score = torch.mean(disconfirm_scores, dim=-1).item()
        if confirm_score > disconfirm_score:
            return True
        else:
            return False

    def compute_parse_text_adapters(self, text: str):
        """Computes the parsed text for the input using intent classifier model.
        """
        anno_intents = self.get_intent_annotations(text)
        anno_slots = self.get_slot_annotations(text)

        do_clarification = False
        decoded_text = ""
        clarification_text = ""
        # NB: if the score is too low, ask for clarification

        if anno_intents[0][1] < 0.50:
            do_clarification = True
            clarification_text = "I'm sorry, I am not sure whether I understood you correctly. Did you mean that you want me to " + \
                                 self.op2clarification[anno_intents[0][0]] + "?"
        best_intent = anno_intents[0][0]

        # discard includes as a separate intent
        # we use it only in combination with others
        if best_intent == "includes":
            best_intent = anno_intents[1][0]
        # remap to "data" because statistic is not in the current grammar
        elif best_intent == "statistic":
            best_intent = "data"
        decoded_text += best_intent

        if best_intent == "model":
            model_slot = None
            for mslot_name in self.model_slots:
                m_slot_values = self.model_slot_words_map[mslot_name]
                for mslot_value in m_slot_values:
                    if mslot_value in text:
                        model_slot = " " + mslot_name
                        break
                if model_slot is not None:
                    break
            if model_slot is None:
                model_slot = " model_summary"
            decoded_text += model_slot
        if best_intent == "data":
            dtype = self.get_data_type(text)
            dflag = self.get_data_flag(text)
            if dflag is not None:
                decoded_text += " " + dtype + "_data_" + dflag

        slot_pattern = self.intent2slot_pattern[best_intent]
        id_adhoc, number_adhoc, token_adhoc = self.check_heuristics(decoded_text, text)
        for slot in slot_pattern:
            decoded_slot_text = ""
            if slot in anno_slots and len(anno_slots[slot]) > 0:
                if slot == "includetoken":
                    decoded_text = "includes and " + decoded_text
                    continue
                if slot == "sent_level":  # we don't need a value in this case
                    decoded_text += " sentence"
                    continue
                try:
                    slot_values = anno_slots[slot]
                except:
                    slot_values = []
                # allow multiple ids
                slot_value = ""
                if len(slot_values) > 1 and slot == "id":
                    prefix = ""
                    for si, slot_value in enumerate(slot_values):
                        prefix += "filter id " + self.clean_up_number(slot_value)
                        if si != len(slot_values) - 1:
                            prefix += " or "
                    decoded_text = prefix + " and " + decoded_text
                    # NB: storing only the last id value
                elif len(slot_values) == 1:
                    slot_value = slot_values[0]
                slot_value = self.clean_up(slot_value)
                if slot in ["id", "number"] and not (slot_value.isdigit()):
                    slot_value = self.clean_up_number(slot_value)
                if slot == "id" and len(slot_value) > 0 and len(slot_values) == 1:
                    decoded_text = "filter id " + str(slot_value) + " and " + decoded_text
                elif slot == "metric" and best_intent == "score":
                    score_setting_parsed = ""
                    slot_value = slot_value.lower()
                    if slot_value in self.score_mapper:
                        slot_value = self.score_mapper[slot_value.lower()]
                    elif not (slot_value in self.scores):
                        slot_value = "default"
                        for s_score in self.scores:
                            if s_score in text.lower():
                                slot_value = s_score
                    for score_setting in self.score_settings:
                        if score_setting in text:
                            score_setting_parsed = " " + score_setting
                    decoded_slot_text += " " + slot_value + score_setting_parsed

                elif len(slot_value) > 0:
                    if slot == "number" and best_intent in self.intent_with_topk_prefix:
                        decoded_slot_text += " topk " + str(slot_value)
                    elif len(slot_values) == 1:  # avoid adding the same id twice
                        decoded_slot_text += " " + str(slot_value)

            else:  # check heuristics
                if slot == "id":
                    if id_adhoc != "":
                        decoded_text = "filter id " + str(id_adhoc) + " and " + decoded_text
                elif slot == "number":
                    if number_adhoc != "" and slot in self.core_slots[best_intent]:
                        if best_intent in self.intent_with_topk_prefix:
                            decoded_slot_text += " topk " + str(number_adhoc)
                        else:
                            decoded_slot_text += " " + str(number_adhoc)
            decoded_text += decoded_slot_text
            if best_intent == "important" and slot == "number" and len(decoded_slot_text) == 0 and (
            not "class_names" in anno_slots):
                decoded_text += " all"
            elif best_intent == "nlpattribute" and slot == "number" and len(decoded_slot_text) == 0 and (
            not "sent_level" in anno_slots):
                decoded_text += " all"
            elif best_intent == "keywords" and slot == "number" and len(decoded_slot_text) == 0:
                decoded_text += " all"
            elif best_intent == "score" and slot == "metric" and not ("metric" in anno_slots):
                score_val = " default"
                for score in self.scores:
                    if score in text:
                        score_val = " " + score
                decoded_text += score_val

        return None, decoded_text, do_clarification, clarification_text


def get_f1scores(predicted_and_gold, labels):
    scores = dict()
    labels = [lb.replace(" ", "_") for lb in labels]
    for label in labels:
        scores[label] = {"tp": 0, "fp": 0, "fn": 0}
    match = 0
    total = 0
    for i, pair in enumerate(predicted_and_gold):
        pred = pair[0]
        gold = pair[1]
        pred = pred.replace("mistake count", "mistake_count").replace("mistake sample", "mistake_sample")
        gold = gold.replace("mistake count", "mistake_count").replace("mistake sample", "mistake_sample")
        pred_list = pred.split()
        gold_list = gold.split()
        for el in pred_list:
            if el in labels:
                total += 1
                if el in gold_list:
                    match += 1
                    scores[el]["tp"] += 1
                else:
                    scores[el]["fp"] += 1
        for el in gold_list:
            if el in labels and not (el in pred_list):
                scores[el]["fn"] += 1
    intent_accuracy = round(match / total, 3)

    micro_prec = 0
    micro_rec = 0
    micro_f1 = 0

    f1scores = 0
    all_tp = 0
    all_fp = 0
    all_fn = 0

    f1_per_label = dict()
    # compute f1 scores (per label)
    for label in labels:
        f1_per_label[label] = {"prec": 0, "rec": 0, "f1": 0}
        tp = scores[label]["tp"]
        fp = scores[label]["fp"]
        fn = scores[label]["fn"]
        all_tp += tp
        all_fp += fp
        all_fn += fn
        if (tp + fp) > 0:
            prec = tp / (tp + fp)
        else:
            prec = 0
        if (tp + fn) > 0:
            rec = tp / (tp + fn)
        else:
            rec = 0
        if (prec + rec) > 0:
            f1score = 2 * prec * rec / (prec + rec)
        else:
            f1score = 0
        f1scores += f1score
        f1_per_label[label]["prec"] = round(prec, 3)
        f1_per_label[label]["rec"] = round(rec, 3)
        f1_per_label[label]["f1"] = round(f1score, 3)
    if (all_tp + all_fp) > 0:
        micro_prec = all_tp / (all_tp + all_fp)
    if (all_tp + all_fn) > 0:
        micro_rec = all_tp / (all_tp + all_fn)
    if (micro_prec + micro_rec) > 0:
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)

    # compute macro f1 score (avg)
    macro_f1 = round(f1scores / len(labels), 3)
    micro_f1 = round(micro_f1, 3)

    return f1_per_label, macro_f1, micro_f1, intent_accuracy


def evaluate(parser, val_data_path):
    user_parsed_tuples = []
    with open(val_data_path) as f:
        lines = f.readlines()
        lines = [ln for ln in lines if len(ln.strip()) > 0]
        for i in range(0, len(lines), 2):
            user_input = lines[i].strip()
            parse_output = lines[i + 1].replace("[E]", "").strip()
            user_parsed_tuples.append((user_input, parse_output))
    total = 0
    matched = 0
    predicted_and_gold = []
    for i in range(len(user_parsed_tuples)):
        input_text = user_parsed_tuples[i][0]
        _, decoded, _, _ = parser.compute_parse_text_adapters(input_text.strip())
        gold_parse = user_parsed_tuples[i][1]

        if val_data_path.startswith("user"):
            decoded += " [e]"

        predicted_and_gold.append((decoded, gold_parse))
        print("input: " + input_text)
        print("gold: " + gold_parse + " >>> decoded: " + decoded)
        print()
        if decoded == gold_parse:
            matched += 1
        total += 1

    f1_per_label, macro_f1, micro_f1, intent_accuracy = get_f1scores(predicted_and_gold, parser.all_intents)
    accuracy = round(matched / total, 4)
    return f1_per_label, macro_f1, micro_f1, intent_accuracy, accuracy


def main():
    parser = AdapterParse()
    val_data_path = f"./experiments/parsing_interrolang_dev/user_set_interrolang_daily_dialog.txt"
    f1_per_label, macro_f1, micro_f1, intent_accuracy, accuracy = evaluate(parser, val_data_path)
    print("Exact Match Accuracy:", accuracy)
    print("Intent Accuracy:", intent_accuracy)
    print("Micro F1:", micro_f1)
    print("Macro F1:", macro_f1)
    print("F1 Scores for Intents:")
    for intent in sorted(f1_per_label.keys()):
        print(intent, f1_per_label[intent])


if __name__ == "__main__":
    main()
