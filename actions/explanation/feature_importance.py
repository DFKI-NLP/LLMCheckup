import json

import torch
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from explained_models.Explainer.explainer import Explainer
import numpy as np

from explained_models.ModelABC.DANetwork import DANetwork
from explained_models.Tokenizer.tokenizer import HFTokenizer
from actions.custom_input import generate_explanation

from nltk.tokenize import sent_tokenize

from timeout import timeout


def handle_input(parse_text):
    """
    Handle the parse text and return the list of numbers(ids) and topk value if given
    Args:
        parse_text: parse_text from bot

    Returns: id_list, topk

    """
    id_list = []
    topk = None

    for item in parse_text:
        try:
            if int(item):
                if int(item) > 0:
                    id_list.append(int(item))
        except:
            pass

    if "topk" in parse_text:
        if len(id_list) >= 1:
            topk = id_list[-1]

        # filter id 5 or filter id 151 or filter id 315 and nlpattribute topk 10 [E]
        if len(id_list) > 1:
            return id_list[:-1], topk
        else:
            # nlpattribute topk 3 [E]
            return None, topk
    else:
        if len(id_list) >= 1:
            # filter id 213 and nlpattribute all [E]
            if "all" in parse_text:
                return id_list, None

            # filter id 213 and nlpattribute sentence [E]
            # if "sentence" in parse_text:
            #     return id_list, -1

        # nlpattribute [E]
        return id_list, topk


# def get_explanation(dataset_name, inputs, conversation, file_name="sentence_level"):
#     """
#     Get explanation list
#     Args:
#         conversation:
#         dataset_name: dataset name
#         inputs: list of inputs
#         file_name: cache file name
#
#     Returns:
#         res_list: results in list
#     """
#     if dataset_name == "boolq":
#         model = conversation.get_var("model").contents.model
#     elif dataset_name == "daily_dialog":
#         model = conversation.get_var("model").contents
#     elif dataset_name == "olid":
#         model = conversation.get_var("model").contents.model
#     else:
#         raise NotImplementedError(f"The dataset {dataset_name} is not supported!")
#
#     res_list = generate_explanation(model, dataset_name, inputs, conversation, file_name=file_name)
#
#     return res_list


# def get_visualization(attr, topk, original_text, conversation):
#     """
#     Get visualization on given input
#     Args:
#         attr: attribution list
#         topk: top k value
#         original_text: original text
#         conversation: conversation object
#
#     Returns:
#         heatmap in html form
#     """
#     return_s = ""
#
#     # Get indices according to absolute attribution scores ascending
#     idx = np.argsort(np.absolute(np.copy(attr)))
#
#     # Get topk tokens
#     topk_tokens = []
#     for i in np.argsort(attr)[-topk:][::-1]:
#         topk_tokens.append(original_text[i])
#
#     score_ranking = []
#     for i in range(len(idx)):
#         score_ranking.append(list(idx).index(i))
#     fraction = 1.0 / (len(original_text) - 1)
#
#     return_s += f"Top {topk} token(s): "
#     for i in topk_tokens:
#         return_s += f"<b>{i}</b>"
#         return_s += " "
#     return_s += '<br>'
#
#     return_s += "<details><summary>"
#     return_s += "The visualization: "
#     return_s += "</summary>"
#     # for i in range(1, len(text_list) - 1):
#     for i in range(1, len(original_text) - 1):
#         if attr[i] >= 0:
#             # Assign red to tokens with positive attribution
#             return_s += f"<span style='background-color:rgba(255,0,0,{round(fraction * score_ranking[i], conversation.rounding_precision)});padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 2; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone'>"
#         else:
#             # Assign blue to tokens with negative attribution
#             return_s += f"<span style='background-color:rgba(0,0,255,{round(fraction * score_ranking[i], conversation.rounding_precision)});padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 2; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone'>"
#         # return_s += text_list[i]
#         return_s += original_text[i]
#         return_s += "</span>"
#         return_s += ' '
#     return_s += "</details>"
#     return_s += '<br><br><br>'
#
#     return return_s

def get_visualization(attr, topk, original_text):
    """
    Get visualization on given input
    Args:
        attr: attribution list
        topk: top k value
        original_text: original text
        conversation: conversation object

    Returns:
        heatmap in html form
    """
    return_s = ""

    # Get indices according to absolute attribution scores ascending
    idx = np.argsort(np.absolute(np.copy(attr)))

    # Get topk tokens
    topk_tokens = []
    for i in np.argsort(attr)[-topk:][::-1]:
        topk_tokens.append(original_text[i])

    score_ranking = []
    for i in range(len(idx)):
        score_ranking.append(list(idx).index(i))
    fraction = 1.0 / (len(original_text) - 1)

    return_s += f"Top {topk} token(s): "
    for i in topk_tokens:
        return_s += f"<b>{i}</b>"
        return_s += " "
    return_s += '<br>'

    return_s += "<details><summary>"
    return_s += "The visualization: "
    return_s += "</summary>"
    # for i in range(1, len(text_list) - 1):
    for i in range(len(original_text)):
        if attr[i] >= 0:
            # Assign red to tokens with positive attribution
            return_s += f"<span style='background-color:rgba(255,0,0,{round(fraction * score_ranking[i], 2)});padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 2; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone'>"
        else:
            # Assign blue to tokens with negative attribution
            return_s += f"<span style='background-color:rgba(0,0,255,{round(fraction * score_ranking[i], 2)});padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 2; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone'>"
        # return_s += text_list[i]
        return_s += original_text[i]
        return_s += "</span>"
        return_s += ' '
    return_s += "</details>"
    return_s += '<br><br><br>'

    return return_s


# def explanation_with_custom_input(conversation, topk):
#     """
#     Get explanation of custom inputs from users
#     Args:
#         conversation: conversation object
#         topk: most top k important tokens
#
#     Returns:
#         formatted string
#     """
#
#     inputs = [conversation.custom_input]
#
#     if len(inputs) == 0:
#         return None
#
#     dataset_name = conversation.describe.get_dataset_name()
#
#     res_list = get_explanation(dataset_name, inputs, conversation)
#     return_s = ""
#     for res in res_list:
#         original_text = res["text"]
#         _input = res["original_text"]
#
#         return_s += "The original text is:  <br>"
#         return_s += "<i>"
#         return_s += _input
#         return_s += "</i>"
#         return_s += "<br><br>"
#
#         attr = res["attributions"]
#
#         assert len(attr) == len(original_text)
#
#     returned_string = get_visualization(attr, topk, original_text, conversation)
#     return_s += returned_string
#     return return_s


# def get_sentence_level_feature_importance(conversation, sentences, simulation):
#     """
#     Sentence level feature importance
#     Args:
#         simulation:
#         conversation: conversation object
#         sentences: A string containing multiple (optionally) sentences
#
#     """
#     # sentences = parse_text[i+1]
#     inputs = sent_tokenize(sentences)
#     dataset_name = conversation.describe.get_dataset_name()
#     res_list = get_explanation(dataset_name, inputs, conversation, file_name="sentence_level")
#
#     return_s = f'The original text is: <i>{sentences}</i> <br><br>'
#     counter = 1
#
#     for res in res_list:
#         attr = res["attributions"]
#
#         text = res["original_text"]
#
#         return_s += "<ul>"
#         return_s += "<li>"
#         return_s += f"Sentence {counter}: <i>{text}</i>"
#         return_s += "</li>"
#
#         return_s += "<li>"
#         return_s += f"Average saliency score: <b>{round(sum(attr) / len(attr), conversation.rounding_precision)}</b>"
#         return_s += "</li>"
#
#         return_s += "<li>"
#         if not simulation:
#             return_s += f"Prediction: <span style=\"background-color: #6CB4EE\">{conversation.class_names[res['predictions']]}</span>"
#         return_s += "</li>"
#         return_s += "</ul>"
#         counter += 1
#     return return_s


# def get_text_by_id(_id, conversation):
#     """
#     Get text with corresponding id
#     Args:
#         _id: id
#         conversation:
#
#     Returns:
#         text string
#     """
#     dataset_name = conversation.describe.get_dataset_name()
#     dataset = conversation.temp_dataset.contents["X"]
#
#     text = ""
#
#     if dataset_name == "boolq":
#
#         text += f"<b>Question: </b>{dataset['question'][_id]} <br>"
#         text += f"<b>Passage: </b>{dataset['passage'][_id]} <br>"
#         original_text = dataset['question'][_id] + " " + dataset['passage'][_id]
#     elif dataset_name == "olid":
#         text += f"<b>Text: </b>{dataset['text'][_id]} <br>"
#         original_text = dataset['text'][_id]
#     elif dataset_name == "daily_dialog":
#         text += f"<b>Dialog: </b>{dataset['dialog'][_id]} <br>"
#         original_text = dataset['dialog'][_id]
#     else:
#         raise NotImplementedError(f"{dataset_name} is not supported!")
#
#     text += "<br>"
#
#     return text, original_text


# def get_attr_by_id(conversation, _id):
#     """
#     Get attribution list and input ids
#     Args:
#         conversation
#         _id
#
#     Returns:
#
#     """
#     dataset_name = conversation.describe.get_dataset_name()
#     data_path = f"./cache/{dataset_name}/ig_explainer_{dataset_name}_explanation.json"
#     fileObject = open(data_path, "r")
#     jsonContent = fileObject.read()
#     json_list = json.loads(jsonContent)
#
#     attr = json_list[_id]["attributions"]
#     input_ids = json_list[_id]["input_ids"]
#
#     return attr, input_ids


# @timeout(60)
# def feature_attribution_with_id(conversation, topk, id_list):
#     # Get the dataset name
#     name = conversation.describe.get_dataset_name()
#
#     data_path = f"./cache/{name}/ig_explainer_{name}_explanation.json"
#     fileObject = open(data_path, "r")
#     jsonContent = fileObject.read()
#     json_list = json.loads(jsonContent)
#
#     if name == 'boolq':
#         tokenizer = conversation.get_var("model").contents.tokenizer
#     elif name == "daily_dialog":
#         tokenizer = HFTokenizer('bert-base-uncased', mode='bert').tokenizer
#     else:
#         tokenizer = conversation.get_var("model").contents.tokenizer
#
#     if topk >= len(json_list[0]["input_ids"]):
#         return "Entered topk is larger than input max length", 1
#     else:
#         if len(id_list) == 1:
#             return_s, original_text = get_text_by_id(id_list[0], conversation)
#
#             attr, input_ids = get_attr_by_id(conversation, id_list[0])
#             converted_text = tokenizer.convert_ids_to_tokens(input_ids)
#             try:
#                 idx = converted_text.index("[PAD]")
#                 return_s += get_visualization(attr[:idx], topk, converted_text[:idx], conversation)
#             except ValueError:
#                 return_s += get_visualization(attr, topk, converted_text, conversation)
#         else:
#             return_s = ""
#             for num in id_list:
#                 return_s += f"For id {num}: <br>"
#                 text, original_text = get_text_by_id(num, conversation)
#                 return_s += text
#
#                 attr, input_ids = get_attr_by_id(conversation, num)
#                 converted_text = tokenizer.convert_ids_to_tokens(input_ids)
#                 try:
#                     idx = converted_text.index("[PAD]")
#                     return_s += get_visualization(attr[:idx], topk, converted_text[:idx], conversation)
#                 except ValueError:
#                     return_s += get_visualization(attr, topk, converted_text, conversation)
#
#                 return_s += "<br>"
#     return return_s


def feature_importance_operation(conversation, parse_text, i, simulation, **kwargs):
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
    # filter id 5 or filter id 151 or filter id 315 and nlpattribute topk 10 [E]
    # filter id 213 and nlpattribute all [E]
    # filter id 33 and nlpattribute topk 1 [E]

    id_list, topk = handle_input(parse_text)

    if topk is None:
        topk = 5

    model = conversation.decoder.gpt_model
    tokenizer = conversation.decoder.gpt_tokenizer

    model_name = conversation.decoder.parser_name
    if model_name == 'EleutherAI/gpt-neo-2.7B':
        fileObject = open("./cache/gpt-neo-2.7b_feature_attribution.json", "r")
    elif model_name == "EleutherAI/gpt-j-6b":
        fileObject = open("./cache/gpt-j-6b_feature_attribution.json", "r")
    else:
        raise NotImplementedError(f"Model {model_name} is unknown!")

    jsonContent = fileObject.read()
    json_list = json.loads(jsonContent)

    # if conversation.used is False and conversation.custom_input is not None:
    #     if "sentence" in parse_text:
    #         return_s = get_sentence_level_feature_importance(conversation, conversation.custom_input, simulation)
    #         return return_s, 1
    #     else:
    #         explanation = explanation_with_custom_input(conversation, topk)
    #         return explanation, 1

    # if topk == -1:
    #     return_s = ''
    #     for _id in id_list:
    #         texts = conversation.temp_dataset.contents['X']
    #         filtered_text = ''
    #
    #         dataset_name = conversation.describe.get_dataset_name()
    #         if dataset_name == "boolq":
    #             filtered_text += texts['question'][_id] + " " + texts['passage'][_id]
    #         elif dataset_name == "olid":
    #             filtered_text = texts['text'][_id]
    #         elif dataset_name == "daily_dialog":
    #             filtered_text = texts['dialog'][_id]
    #
    #         return_s += f'ID {_id}: '
    #         return_s += get_sentence_level_feature_importance(conversation, filtered_text, simulation)
    #         return_s += '<br><br>'
    #     return return_s, 1
    return_s = ""
    for _id in id_list:
        attribution = json_list[_id]["attribution"]
        texts = json_list[_id]["text"]
        decoded_text = [tokenizer.convert_tokens_to_string(text) for text in texts]
        return_s += get_visualization(attribution, topk, decoded_text)
        return_s += "<br><br>"
    return return_s, 1


# class FeatureAttributionExplainer(Explainer):
#     def __init__(self, model, device):
#         super(Explainer).__init__()
#         self.device = device
#         self.model = model.model.to(device)
#         # self.tokenizer = model.tokenizer
#         self.dataloader = model.dataloader
#         self.forward_func = self.get_forward_func()
#         self.explainer = LayerIntegratedGradients(forward_func=self.forward_func,
#                                                   layer=self.get_embedding_layer())
#         self.pad_token_id = self.model.tokenizer.pad_token_id
#
#     def get_forward_func(self):
#         # TODO: Implement forward functions for non-BERT models (LSTM, ...)
#         def bert_forward(input_ids, attention_mask):
#             input_model = {
#                 'input_ids': input_ids.long(),
#                 'attention_mask': attention_mask.long(),
#             }
#             output_model = self.model(**input_model)[0]
#             return output_model
#
#         return bert_forward
#
#     def get_embedding_layer(self):
#         return self.model.base_model.embeddings
#
#     @staticmethod
#     def get_inputs_and_additional_args(base_model, batch):
#         assert 'input_ids' in batch, f'Input ids expected for {base_model} but not found.'
#         assert 'attention_mask' in batch, f'Attention mask expected for {base_model} but not found.'
#         input_ids = batch['input_ids']
#         additional_forward_args = (batch['attention_mask'])
#         return input_ids, additional_forward_args
#
#     def to(self, device):
#         self.device = device
#         self.model.to(self.device)
#
#     def get_baseline(self, batch):
#         assert 'special_tokens_mask' in batch
#         if self.pad_token_id == 0:
#             # all non-special token ids are replaced by 0, the pad id
#             baseline = batch['input_ids'] * batch['special_tokens_mask']
#             return baseline
#         else:
#             baseline = batch['input_ids'] * batch['special_tokens_mask']  # all input ids now 0
#             # add pad_id everywhere,
#             # substract again where special tokens are, leaves non special tokens with pad id
#             # and conserves original pad ids
#             baseline = (baseline + self.pad_token_id) - (batch['special_tokens_mask'] * self.pad_token_id)
#             return baseline
#
#     def compute_feature_attribution_scores(
#             self,
#             batch
#     ):
#         r"""
#         :param batch
#         :return:
#         """
#         self.model.eval()
#         self.model.zero_grad()
#         batch = {k: v.to(self.device) for k, v in batch.items()}
#         inputs, additional_forward_args = self.get_inputs_and_additional_args(
#             base_model=type(self.model.base_model),
#             batch=batch
#         )
#         predictions = self.forward_func(
#             inputs,
#             *additional_forward_args
#         )
#         pred_id = torch.argmax(predictions, dim=1)
#         baseline = self.get_baseline(batch=batch)
#         attributions = self.explainer.attribute(
#             inputs=inputs,
#             n_steps=50,
#             additional_forward_args=additional_forward_args,
#             target=pred_id,
#             baselines=baseline,
#             internal_batch_size=1,
#         )
#         attributions = torch.sum(attributions, dim=2)
#         return attributions, predictions
#
#     def generate_explanation(self, store_data=False, data_path="../../cache/boolq/ig_explainer_boolq_explanation.json"):
#         def detach_to_list(t):
#             return t.detach().cpu().numpy().tolist() if type(t) == torch.Tensor else t
#
#         if store_data:
#             json_list = []
#
#         for idx_batch, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), position=0, leave=True):
#             if idx_batch % 1000 == 0:
#                 print(f'(Progress) Processing batch {idx_batch} / instance {idx_batch * self.dataloader.batch_size}')
#             attribution, predictions = self.compute_feature_attribution_scores(batch)
#
#             for idx_instance in range(len(batch['input_ids'])):
#                 idx_instance_running = (idx_batch * self.dataloader.batch_size)
#
#                 ids = detach_to_list(batch['input_ids'][idx_instance])
#                 label = detach_to_list(batch['labels'][idx_instance])
#                 attrbs = detach_to_list(attribution[idx_instance])
#                 preds = detach_to_list(predictions[idx_instance])
#                 result = {'batch': idx_batch,
#                           'instance': idx_instance,
#                           'index_running': idx_instance_running,
#                           'input_ids': ids,
#                           'label': label,
#                           'attributions': attrbs,
#                           'predictions': preds}
#                 if store_data:
#                     json_list.append(result)
#         if store_data:
#             jsonString = json.dumps(json_list)
#             jsonFile = open(data_path, "w")
#             jsonFile.write(jsonString)
#             jsonFile.close()
