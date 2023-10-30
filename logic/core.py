"""The main script that controls conversation logic.

This file contains the core logic for facilitating conversations. It orchestrates the necessary
routines for setting up conversations, controlling the state of the conversation, and running
the functions to get the responses to user inputs.
"""

import gin
import numpy as np
import os
import re
import pickle
import secrets
import sys
import torch
import random
from flask import Flask
from random import seed as py_random_seed

from explained_models.ModelABC.DANetwork import DANetwork
from word2number import w2n
import string

from logic.action import run_action
from logic.conversation import Conversation
from logic.decoder import Decoder
from logic.parser import Parser, get_parse_tree
from logic.prompts import Prompts
from logic.utils import read_and_format_data, read_precomputed_prediction
from logic.write_to_log import log_dialogue_input
from logic.transformers import TransformerModel

from sentence_transformers import SentenceTransformer, util

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)


@gin.configurable
def load_sklearn_model(filepath):
    """Loads a sklearn model."""
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model


@gin.configurable
def load_hf_model(model_id, name):
    """ Loads a (local) Hugging Face model from a directory containing a pytorch_model.bin file and a config.json file.
    """
    return TransformerModel(model_id, name)
    # transformers.AutoModel.from_pretrained(model_id)


@gin.configurable
class ExplainBot:
    """The ExplainBot Class."""

    def __init__(self,
                 # model_file_path: str,
                 dataset_file_path: str,
                 background_dataset_file_path: str,
                 dataset_index_column: int,
                 target_variable_name: str,
                 categorical_features: list[str],
                 numerical_features: list[str],
                 remove_underscores: bool,
                 name: str,
                 text_fields: list[str],
                 parsing_model_name: str = "ucinlp/diabetes-t5-small",
                 seed: int = 0,
                 prompt_metric: str = "cosine",
                 prompt_ordering: str = "ascending",
                 t5_config: str = None,
                 use_guided_decoding: bool = True,
                 feature_definitions: dict = None,
                 skip_prompts: bool = False,
                 simulation: bool = False
                ):
        """The init routine.

        Arguments:
            model_file_path: The filepath of the **user provided** model to logic. This model
                             should end with .pkl and support sklearn style functions like
                             .predict(...) and .predict_proba(...)
            dataset_file_path: The path to the dataset used in the conversation. Users will understand
                               the model's predictions on this dataset.
            background_dataset_file_path: The path to the dataset used for the 'background' data
                                          in the explanations.
            dataset_index_column: The index column in the data. This is used when calling
                                  pd.read_csv(..., index_col=dataset_index_column)
            target_variable_name: The name of the column in the dataset corresponding to the target,
                                  i.e., 'y'
            categorical_features: The names of the categorical features in the data. If None, they
                                  will be guessed.
            numerical_features: The names of the numeric features in the data. If None, they will
                                be guessed.
            remove_underscores: Whether to remove underscores in the feature names. This might help
                                performance a bit.
            name: The dataset name
            parsing_model_name: The name of the parsing model. See decoder.py for more details about
                                the allowed models.
            seed: The seed
            prompt_metric: The metric used to compute the nearest neighbor prompts. The supported options
                           are cosine, euclidean, and random
            prompt_ordering:
            t5_config: The path to the configuration file for t5 models, if using one of these.
            skip_prompts: Whether to skip prompt generation. This is mostly useful for running fine-tuned
                          models where generating prompts is not necessary.
        """
        super(ExplainBot, self).__init__()
        # Set seeds
        np.random.seed(seed)
        py_random_seed(seed)
        torch.manual_seed(seed)

        self.bot_name = name

        # Prompt settings
        self.prompt_metric = prompt_metric
        self.prompt_ordering = prompt_ordering
        self.use_guided_decoding = use_guided_decoding

        # A variable used to help file uploads
        self.manual_var_filename = None

        self.decoding_model_name = parsing_model_name

        # Initialize completion + parsing modules
        app.logger.info(f"Loading parsing model {parsing_model_name}...")
        self.decoder = Decoder(parsing_model_name,
                               t5_config,
                               use_guided_decoding=self.use_guided_decoding, dataset_name=name)

        # Initialize parser + prompts as None
        # These are done when the dataset is loaded
        self.prompts = None
        self.parser = None

        # Add text fields, e.g. "question" and "passage" for BoolQ
        self.text_fields = text_fields

        # Set up the conversation object
        self.conversation = Conversation(eval_file_path=dataset_file_path,
                                         feature_definitions=feature_definitions,
                                         decoder=self.decoder,
                                         text_fields=self.text_fields)

        # Load the model into the conversation
        self.load_model()

        # Load the dataset into the conversation
        self.load_dataset(dataset_file_path,
                          dataset_index_column,
                          target_variable_name,
                          categorical_features,
                          numerical_features,
                          remove_underscores,
                          store_to_conversation=True,
                          skip_prompts=skip_prompts)

        self.parsed_text = None
        self.user_text = None
        
        self.simulation = simulation

        if "adapters" in parsing_model_name:
            from transformers import AutoAdapterModel, AutoTokenizer
            from transformers import TextClassificationPipeline, TokenClassificationPipeline

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

            self.all_intents = ["adversarial", "augment", "includes", "cfe", "similar", "predict", "self", "data",
                                "show", "likelihood", "model", "function", "score", "countdata", "label",
                                "mistake count", "mistake sample", "keywords", "nlpattribute", "rationalize",
                                "important", "statistic", "randompredict"]
            self.id2label_str = dict()
            for i, intent_name in enumerate(self.all_intents):
                self.id2label_str[i] = intent_name
            # Mapping slots to the fixed values for scores (as defined in grammar.py)
            self.score_mapper = {"positive predictive value": "ppv", "negative predictive value": "npv",
                                 "acc": "accuracy", "prec": "precision", "rec": "recall", "f 1": "f1"}
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
        data_name = ["inform me test data name", "name of training data", "how is the test set called?", "what's the name of the data?"]
        data_source = ["where does training data come from", "where do you get test data", "the source of the dataset?"]
        data_language = ["show me the language of training data", "language of training data", "tell me the language of testing data", "what's the language of the model?"]
        data_number = ["how many training data is used", "count test data points", "tell me the number of data points"]
        thanks = ["Thanks!", "OK!", "I see", "Thanks a lot!", "Thank you.", "Alright, thank you!", "That's nice, thanks a lot :)", "Good, thanks!", "Thank you very much.", "Looks good, thank you!", "Great, thank you very much!", "Ok, thanks!", "Thank you for the answer."]
        bye = ["Goodbye!", "Bye-bye!", "Bye!", "Ok, bye then!", "That's all, bye!", "See you next time!", "Thanks for the chat, bye!"]

        self.dialogue_flow_map = {"thanks": ["You are welcome!", "No problem.", "I'm glad I could help.", "Can I help you with something else?", "Is there anything else I could do for you?"], "bye": ["Goodbye!", "Bye-bye!", "Have a nice day!", "See you next time!"], "sorry": ["Sorry! I couldn't understand that. Could you please try to rephrase?", "My apologies, I did not get what you mean.", "I'm sorry but could you rephrase the message, please?", "I'm not sure I can do this. Maybe you have another request for me?", "This is likely out of my expertise, can I help you with something else?", "This was a bit unclear. Could you rephrase it, please?", "Let's try another option. I'm afraid I don't have an answer for this."]}

        # Compute embedding for data flags
        self.data_name = self.st_model.encode(data_name, convert_to_tensor=True)
        self.data_source = self.st_model.encode(data_source, convert_to_tensor=True)
        self.data_language = self.st_model.encode(data_language, convert_to_tensor=True)
        self.data_number = self.st_model.encode(data_number, convert_to_tensor=True)

        # Compute embeddings for confirm/disconfirm
        self.confirm = self.st_model.encode(confirm, convert_to_tensor=True)
        self.disconfirm = self.st_model.encode(disconfirm, convert_to_tensor=True)

        # Compute embeddings for thanks/bye
        self.thanks = self.st_model.encode(thanks, convert_to_tensor=True)
        self.bye = self.st_model.encode(bye, convert_to_tensor=True)


    def get_data_type(self, text:str):
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

    def init_loaded_var(self, name: bytes):
        """Inits a var from manual load."""
        self.manual_var_filename = name.decode("utf-8")

    def load_model(self):
        """Loads a model.

        This routine loads a model into the conversation
        from a specified file path. The model will be saved as a variable
        names 'model' in the conversation, overwriting an existing model.

        The routine determines the type of model from the file extension.
        Scikit learn models should be saved as .pkl's and torch as .pt.

        Arguments:
            filepath: the filepath of the model.
        Returns:
            success: whether the model was saved successfully.
        """
        app.logger.info(f"Loading inference model...")

        class Model:
            def predict(self, data, text, conversation=None):
                """
                Arguments:
                    data: Pandas DataFrame containing columns of text data
                    text: preprocessed parse_text
                    conversation:
                """
                str2int = {"offensive": 1, "non-offensive": 0}

                json_list = read_precomputed_prediction(conversation)

                # Get indices of dataset to filter json_list with
                if data is not None:
                    data_indices = data.index.to_list()

                if text is None:
                    temp = []
                    for item in json_list:
                        if item["idx"] in data_indices:
                            temp.append(str2int[item["prediction"]])

                    return np.array(temp)
                else:
                    res = list([str2int[json_list[text]["prediction"]]])
                    return np.array(res)
        model = Model()
        self.conversation.add_var('model', model, 'model')
        app.logger.info("...done")
        return 'success'

    def load_dataset(self,
                     filepath: str,
                     index_col: int,
                     target_var_name: str,
                     cat_features: list[str],
                     num_features: list[str],
                     remove_underscores: bool,
                     store_to_conversation: bool,
                     skip_prompts: bool = False):
        """Loads a dataset, creating parser and prompts.

        This routine loads a dataset. From this dataset, the parser
        is created, using the feature names, feature values to create
        the grammar used by the parser. It also generates prompts for
        this particular dataset, to be used when determine outputs
        from the model.

        Arguments:
            filepath: The filepath of the dataset.
            index_col: The index column in the dataset
            target_var_name: The target column in the data, i.e., 'y' for instance
            cat_features: The categorical features in the data
            num_features: The numeric features in the data
            remove_underscores: Whether to remove underscores from feature names
            store_to_conversation: Whether to store the dataset to the conversation.
            skip_prompts: whether to skip prompt generation.
        Returns:
            success: Returns success if completed and store_to_conversation is set to true. Otherwise,
                     returns the dataset.
        """
        app.logger.info(f"Loading dataset at path {filepath}...")

        # Read the dataset and get categorical and numerical features
        dataset, y_values, categorical, numeric = read_and_format_data(filepath,
                                                                       index_col,
                                                                       target_var_name,
                                                                       cat_features,
                                                                       num_features,
                                                                       remove_underscores)

        if store_to_conversation:

            # Store the dataset
            self.conversation.add_dataset(dataset, y_values, categorical, numeric)

            # Set up the parser
            self.parser = Parser(cat_features=categorical,
                                 num_features=numeric,
                                 dataset=dataset,
                                 class_names=self.conversation.class_names)

            # Generate the available prompts
            # make sure to add the "incorrect" temporary feature
            # so we generate prompts for this
            self.prompts = Prompts(cat_features=categorical,
                                   num_features=numeric,
                                   target=np.unique(list(y_values)),
                                   feature_value_dict=self.parser.features,
                                   class_names=self.conversation.class_names,
                                   skip_creating_prompts=skip_prompts)
            self.conversation.prompts = self.prompts
            app.logger.info("..done")

            return "success"
        else:
            return dataset

    def set_num_prompts(self, num_prompts):
        """Updates the number of prompts to a new number"""
        self.prompts.set_num_prompts(num_prompts)

    @staticmethod
    def gen_almost_surely_unique_id(n_bytes: int = 30):
        """To uniquely identify each input, we generate a random 30 byte hex string."""
        return secrets.token_hex(n_bytes)

    @staticmethod
    def log(logging_input: dict):
        """Performs the system logging."""
        assert isinstance(logging_input, dict), "Logging input must be dict"
        assert "time" not in logging_input, "Time field will be added to logging input"
        log_dialogue_input(logging_input)

    @staticmethod
    def build_logging_info(bot_name: str,
                           username: str,
                           response_id: str,
                           system_input: str,
                           parsed_text: str,
                           system_response: str):
        """Builds the logging dictionary."""
        return {
            'bot_name': bot_name,
            'username': username,
            'id': response_id,
            'system_input': system_input,
            'parsed_text': parsed_text,
            'system_response': system_response
        }

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
            app.logger.info(f"value is not a number: {text}")
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

    def check_dialogue_flow_intents(self, text: str):
        """Checks whether the user says thanks/bye etc."""
        # Compute cosine-similarities
        text = self.st_model.encode(text, convert_to_tensor=True)
        thanks_scores = util.cos_sim(text, self.thanks)
        bye_scores = util.cos_sim(text, self.bye)
        max_thanks_score = torch.max(thanks_scores)
        max_bye_score = torch.max(bye_scores)
        if max_thanks_score > max_bye_score and max_thanks_score > 0.50:
            return "thanks"
        elif max_bye_score > 0.50:
            return "bye"
        return None

    def compute_parse_text(self, text: str, error_analysis: bool = False):
        """Computes the parsed text from the user text input.

        Arguments:
            error_analysis: Whether to do an error analysis step, where we compute if the
                            chosen prompts include all the
            text: The text the user provides to the system
        Returns:
            parse_tree: The parse tree from the formal grammar decoded from the user input.
            parse_text: The decoded text in the formal grammar decoded from the user input
                        (Note, this is just the tree in a string representation).
        """
        nn_prompts = None
        if error_analysis:
            grammar, prompted_text, nn_prompts = self.compute_grammar(text, error_analysis=error_analysis)
        else:
            grammar, prompted_text = self.compute_grammar(text, error_analysis=error_analysis)
        app.logger.info("About to decode")
        # Do guided-decoding to get the decoded text
        api_response = self.decoder.complete(
            prompted_text, grammar=grammar)
        decoded_text = api_response['generation']

        app.logger.info(f'Decoded text {decoded_text}')

        # Compute the parse tree from the decoded text
        # NOTE: currently, we're using only the decoded text and not the full
        # tree. If we need to support more complicated parses, we can change this.
        parse_tree, parsed_text = get_parse_tree(decoded_text)
        if error_analysis:
            return parse_tree, parsed_text, nn_prompts
        else:
            return parse_tree, parsed_text,

    def compute_parse_text_t5(self, text: str):
        """Computes the parsed text for the input using a t5 model.

        This supposes the user has finetuned a t5 model on their particular task and there isn't
        a need to do few shot
        """
        grammar, prompted_text = self.compute_grammar(text)
        decoded_text = self.decoder.complete(text, grammar)
        app.logger.info(f"t5 decoded text {decoded_text}")
        parse_tree, parse_text = get_parse_tree(decoded_text[0])
        return parse_tree, parse_text

    def compute_parse_text_adapters(self, text: str):
        """Computes the parsed text for the input using intent classifier model.
        """
        do_clarification = False
        decoded_text = ""
        clarification_text = ""

        anno_intents = self.get_intent_annotations(text)
        anno_slots = self.get_slot_annotations(text)

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
                if slot == "includetoken" and self.conversation.include_word is not None:
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
                    self.conversation.prev_id = str(slot_value)
                elif len(slot_values) == 1:
                    slot_value = slot_values[0]
                slot_value = self.clean_up(slot_value)
                if slot == "includetoken":
                    self.conversation.include_word = self.clean_up(slot_value)
                elif slot in ["id", "number"] and not (slot_value.isdigit()):
                    slot_value = self.clean_up_number(slot_value)
                if slot == "id" and len(slot_value) > 0 and len(slot_values) == 1:
                    decoded_text = "filter id " + str(slot_value) + " and " + decoded_text
                    self.conversation.prev_id = str(slot_value)
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
                    elif len(slot_values) == 1: # avoid adding the same id twice
                        decoded_slot_text += " " + str(slot_value)

            else:  # check heuristics
                if slot == "includetoken" and token_adhoc != "":
                    self.conversation.include_word = token_adhoc
                elif slot == "id":
                    if id_adhoc != "":
                        decoded_text = "filter id " + str(id_adhoc) + " and " + decoded_text
                        self.conversation.prev_id = str(id_adhoc)
                    elif self.conversation.prev_id is not None and (
                            self.has_deictic(text) or slot in self.core_slots[best_intent]) and self.conversation.custom_input is None:
                        decoded_text = "filter id " + self.conversation.prev_id + " and " + decoded_text
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
        self.conversation.store_last_parse(decoded_text)
        app.logger.info(f"adapters decoded text {decoded_text}")
        return None, decoded_text, do_clarification, clarification_text

    def compute_grammar(self, text, error_analysis: bool = False):
        """Computes the grammar from the text.

        Arguments:
            text: the input text
            error_analysis: whether to compute extra information used for error analyses
        Returns:
            grammar: the grammar generated for the input text
            prompted_text: the prompts computed for the input text
            nn_prompts: the knn prompts, without extra information that's added for the full
                        prompted_text provided to prompt based models.
        """
        nn_prompts = None
        app.logger.info("getting prompts")
        # Compute KNN prompts
        if error_analysis:
            prompted_text, adhoc, nn_prompts = self.prompts.get_prompts(text,
                                                                        self.prompt_metric,
                                                                        self.prompt_ordering,
                                                                        error_analysis=error_analysis)
        else:
            prompted_text, adhoc = self.prompts.get_prompts(text,
                                                            self.prompt_metric,
                                                            self.prompt_ordering,
                                                            error_analysis=error_analysis)
        app.logger.info("getting grammar")
        # Compute the formal grammar, making modifications for the current input
        grammar = self.parser.get_grammar(
            adhoc_grammar_updates=adhoc)

        if error_analysis:
            return grammar, prompted_text, nn_prompts
        else:
            return grammar, prompted_text

    def update_state(self, text: str, user_session_conversation: Conversation):
        """The main conversation driver.

        The function controls state updates of the conversation. It accepts the
        user input and ultimately returns the updates to the conversation.

        Arguments:
            text: The input from the user to the conversation.
            user_session_conversation: The conversation sessions for the current user.
        Returns:
            output: The response to the user input.
        """

        if any([text is None, (self.decoding_model_name != "adapters" and self.prompts is None), self.parser is None]):
            return ''

        app.logger.info(f'USER INPUT: {text}')
        do_clarification = False

        # check if we have simply thanks or bye and return corresp. string in this case
        df_intent = self.check_dialogue_flow_intents(text)
        if df_intent is not None:
            parsed_text = df_intent
            returned_item = random.choice(self.dialogue_flow_map[parsed_text])
        else:
            # Parse user input into text abiding by formal grammar
            if self.decoding_model_name == "adapters":
                parse_tree = ""
                if user_session_conversation.needs_clarification:
                    user_session_conversation.needs_clarification = False
                    if self.is_confirmed(text):
                        parsed_text = user_session_conversation.get_last_parse()
                        if len(parsed_text) > 0:
                            parsed_text = parsed_text[-1]
                    else:
                        parse_tree, parsed_text, do_clarification, clarification_text = self.compute_parse_text_adapters(
                            text)
                else:
                    parse_tree, parsed_text, do_clarification, clarification_text = self.compute_parse_text_adapters(text)
                    if do_clarification:
                        user_session_conversation.needs_clarification = True
                        return (clarification_text)
            elif "t5" not in self.decoding_model_name:
                parse_tree, parsed_text = self.compute_parse_text(text)

                if self.decoding_model_name == "mistralai/Mistral-7B-v0.1":
                    ls = parsed_text.split(" ")
                    for (idx, i) in enumerate(ls):
                        if "<s>" in i:
                            ls[idx] = i.split("<s>")[0]
                    ls = [i for i in ls if i != '']
                    parsed_text = " ".join(ls)
                elif self.decoding_model_name == "llama":
                    pass
                else:
                    pass

                app.logger.info(f"parsed text: {parsed_text}")
            else:
                parse_tree, parsed_text = self.compute_parse_text_t5(text)

            if self.decoding_model_name == "adapters" and do_clarification:
                returned_item = parsed_text
            else:
                # Run the action in the conversation corresponding to the formal grammar
                user_session_conversation.needs_clarification = False
                returned_item = run_action(
                    user_session_conversation, parse_tree, parsed_text, self.simulation)
         
        self.parsed_text = parsed_text

        username = user_session_conversation.username

        response_id = self.gen_almost_surely_unique_id()
        logging_info = self.build_logging_info(self.bot_name,
                                               username,
                                               response_id,
                                               text,
                                               parsed_text,
                                               returned_item)
        self.log(logging_info)
        # Concatenate final response, parse, and conversation representation
        # This is done so that we can split both the parse and final
        # response, then present all the data
        final_result = returned_item + f"<>{response_id}"

        return final_result
