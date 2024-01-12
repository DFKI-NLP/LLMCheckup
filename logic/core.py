"""The main script that controls conversation logic.

This file contains the core logic for facilitating conversations. It orchestrates the necessary
routines for setting up conversations, controlling the state of the conversation, and running
the functions to get the responses to user inputs.
"""
import json

import gin
import numpy as np
import os
import secrets
import sys
import torch
import random
from flask import Flask
from random import seed as py_random_seed

from logic.action import run_action
from logic.conversation import Conversation
from logic.decoder import Decoder
from logic.parser import Parser, get_parse_tree
from logic.prompts import Prompts
from logic.utils import read_and_format_data, get_user_questions_and_parsed_texts
from logic.write_to_log import log_dialogue_input
from logic.constants import operations_with_id, deictic_words, confirm, disconfirm, thanks, bye, dialogue_flow_map, \
    user_prompts, valid_operation_names, operation2set, map2suggestion, no_filter_operations, qatutorial

from parsing.multi_prompt.prompting_parser import MultiPromptParser

from sentence_transformers import SentenceTransformer, util

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)


@gin.configurable
class ExplainBot:
    """The ExplainBot Class."""

    def __init__(self,
                 dataset_file_path: str,
                 dataset_index_column: int,
                 target_variable_name: str,
                 categorical_features: list[str],
                 numerical_features: list[str],
                 remove_underscores: bool,
                 name: str,
                 text_fields: list[str],
                 parsing_model_name: str,
                 load_in_4bits: bool,
                 seed: int = 0,
                 prompt_metric: str = "cosine",
                 prompt_ordering: str = "ascending",
                 use_guided_decoding: bool = True,
                 feature_definitions: dict = None,
                 skip_prompts: bool = False,
                 suggestions: bool = False,
                 use_multi_prompt: bool = False,
                 ):
        """The init routine.

        Arguments:
            dataset_file_path: The path to the dataset used in the conversation. Users will understand
                               the model's predictions on this dataset.
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
            skip_prompts: Whether to skip prompt generation. This is mostly useful for running fine-tuned
                          models where generating prompts is not necessary.
            suggestions: Whether we suggest similar operations to the user.
            use_multi_prompt: Whether we use a multi-prompt approach for parsing the user input
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
        self.decoder = Decoder(parsing_model_name, load_in_4bits,
                               use_guided_decoding=self.use_guided_decoding, dataset_name=name)

        # Initialize parser + prompts as None
        # These are done when the dataset is loaded
        self.prompts = None
        self.parser = None

        # Add text fields, e.g. "question" and "passage" for BoolQ
        self.text_fields = text_fields

        # Add suggestions mode
        self.suggestions = suggestions
        self.suggested_operation = None

        # Add dialogue flow map thanks/bye/sorry
        self.dialogue_flow_map = dialogue_flow_map

        # Set up the conversation object
        self.conversation = Conversation(eval_file_path=dataset_file_path,
                                         feature_definitions=feature_definitions,
                                         decoder=self.decoder,
                                         text_fields=self.text_fields)

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

        self.device = 0 if torch.cuda.is_available() else -1
        self.st_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Add multi-prompt parser (if needed)
        self.use_multi_prompt = use_multi_prompt
        if self.use_multi_prompt:
            self.mprompt_parser = MultiPromptParser(self.decoder.gpt_model, self.decoder.gpt_tokenizer, self.st_model,
                                                    self.device)
        else:
            self.mprompt_parser = None

        # Compute embeddings for confirm/disconfirm
        self.confirm = self.st_model.encode(confirm, convert_to_tensor=True)
        self.disconfirm = self.st_model.encode(disconfirm, convert_to_tensor=True)

        # Compute embeddings for thanks/bye
        self.thanks = self.st_model.encode(thanks, convert_to_tensor=True)
        self.bye = self.st_model.encode(bye, convert_to_tensor=True)

        self.user_questions, self.parsed_texts = get_user_questions_and_parsed_texts()

        self.history = []

    def write_to_history(self, user_text, response):
        self.history.append({
            "user_text": user_text,
            "response": response
        })

    def export_history(self):
        jsonString = json.dumps(self.history)
        jsonFile = open(f"./cache/history.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    def has_deictic(self, text):
        for deictic in deictic_words:
            if " " + deictic in text.lower() or deictic + " " in text.lower():
                return True
        return False

    def id_needed(self, parsed_text):
        operation_needs_id = False
        for w in parsed_text.strip().split():
            if w in operations_with_id:
                operation_needs_id = True
                break
        if not (" id " in parsed_text) and operation_needs_id:
            return True
        return False

    def init_loaded_var(self, name: bytes):
        """Inits a var from manual load."""
        self.manual_var_filename = name.decode("utf-8")

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
        if max_thanks_score > max_bye_score and max_thanks_score > 0.5:
            return "thanks"
        elif max_bye_score > 0.5:
            return "bye"
        return None

    def suggestion_confirmed(self, text: str):
        """Checks whether the user confirmed the suggestion"""
        text = self.st_model.encode(text, convert_to_tensor=True)
        confirm_scores = util.cos_sim(text, self.confirm)
        disconfirm_scores = util.cos_sim(text, self.disconfirm)
        confirm_score = torch.mean(confirm_scores, dim=-1).item()
        disconfirm_score = torch.mean(disconfirm_scores, dim=-1).item()
        if confirm_score > disconfirm_score:
            return True, torch.max(confirm_scores.flatten()).item()
        return False, torch.max(disconfirm_scores.flatten()).item()

    def remove_filter_if_needed(self, suggested_operation: str, selected_operation: str):
        if (selected_operation) in no_filter_operations:
            return selected_operation + " [e]"
        return suggested_operation

    def pick_relevant_operation(self, parsed_text: str):
        suggested_operation = None
        suggestion_text = ""
        op_words = []
        for word in parsed_text.split():
            if word in valid_operation_names:
                op_words.append(word)
        parsed_text_operation = " ".join(op_words)
        if len(parsed_text_operation) > 0:
            parsed_text_operation = parsed_text_operation
            selected_operation = random.choice(
                [op for op in operation2set[parsed_text_operation] if op != parsed_text_operation])
            suggested_operation = parsed_text.replace(parsed_text_operation, selected_operation)
            suggested_operation = self.remove_filter_if_needed(suggested_operation, selected_operation)
            for qa_op in qatutorial:
                if suggested_operation.startswith(qa_op):
                    suggested_operation = "qatutorial " + suggested_operation
                    break
            suggestion_text = random.choice(map2suggestion[selected_operation])
        # check whether the user already asked about this operation
        # or we suggested it earlier
        if suggested_operation in self.conversation.previous_operations:
            suggested_operation = None
            suggestion_text = ""
        if len(suggestion_text) > 0:
            suggestion_text = "<br><b>Follow-up:</b><br><div>" + suggestion_text + "</div>"
        return suggested_operation, suggestion_text

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
        if self.use_multi_prompt:
            parsed_text = self.mprompt_parser.parse_user_input(text)
            parse_tree = None
        else:
            # Do guided-decoding to get the decoded text
            api_response = self.decoder.complete(
                prompted_text, grammar=grammar)
            decoded_text = api_response['generation']

            # post process the parsed text for llama/mistral
            ls = decoded_text.split(" ")
            for (idx, i) in enumerate(ls):
                if "<s>" in i:
                    ls[idx] = i.split("<s>")[0]
            ls = [i for i in ls if i != '']
            decoded_text = " ".join(ls)

            app.logger.info(f'Decoded text {decoded_text}')

            # Compute the parse tree from the decoded text
            # NOTE: currently, we're using only the decoded text and not the full
            # tree. If we need to support more complicated parses, we can change this.
            parse_tree, parsed_text = get_parse_tree(decoded_text)

        if self.id_needed(parsed_text): #and self.has_deictic(text):
            if self.conversation.custom_input is None and self.conversation.prev_id is not None:
                parsed_text = "filter id " + str(self.conversation.prev_id) + " and " + parsed_text
        # store the previous id value
        current_id = None
        parsed_words = parsed_text.strip().split()
        for w_i, w in enumerate(parsed_words):
            if w == "id":
                current_id = parsed_words[w_i + 1]
        if not (current_id is None):
            self.conversation.prev_id = current_id

        if error_analysis:
            return parse_tree, parsed_text, nn_prompts
        else:
            return parse_tree, parsed_text,

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

    def check_prompt_availability(self, user_question):
        """
        Check if the user question is contained in our predefined prompt set
        :param text: user question
        :return: availability and idx
        """

        def compare_str(str1, str2):
            """

            :param str1: original string
            :param str2: predefined user question
            :return:
            """

            str1 = str1.lower().split()
            str2 = str2.lower().split()

            idx_list = []
            if len(str1) == len(str2):
                counter = 0
                for j in range(len(str1)):
                    if counter <= 2:
                        if str1[j] != str2[j]:
                            idx_list.append((j, str2[j]))
                            counter += 1
                    else:
                        return False, []
                if counter <= 2 and str1[0].lower() == str2[0].lower():
                    return True, idx_list
            else:
                return False, []

        flag = False
        idx = None

        for (i, item) in enumerate(self.user_questions):
            try:
                flag, idx_list = compare_str(item, user_question)
            except TypeError:
                pass

            if flag:
                return flag, i, idx_list

        return flag, idx, []

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
        if self.suggestions and not (self.suggested_operation is None):
            # check if the user agreed to suggestion
            suggestion_confirmed, max_response_match = self.suggestion_confirmed(text)
            username = user_session_conversation.username
            response_id = self.gen_almost_surely_unique_id()
            if suggestion_confirmed and max_response_match >= 0.5:
                returned_item = run_action(
                    user_session_conversation, None, self.suggested_operation)
                logging_info = self.build_logging_info(self.bot_name,
                                                       username,
                                                       response_id,
                                                       text,
                                                       self.suggested_operation,
                                                       returned_item)
                self.log(logging_info)
                self.conversation.previous_operations.append(self.suggested_operation)

                final_result = returned_item + f"<>{response_id}"
                self.suggested_operation = None
                return final_result
            elif max_response_match >= 0.5:
                returned_item = random.choice(user_prompts)
                final_result = returned_item + f"<>{response_id}"
                self.suggested_operation = None
                return final_result

            # reset the suggestions mode
            self.suggested_operation = None

        if any([text is None, self.prompts is None, self.parser is None]):
            return ''

        app.logger.info(f'USER INPUT: {text}')
        self.conversation.user_input = text

        # check if we have simply thanks or bye and return corresp. string in this case
        df_intent = self.check_dialogue_flow_intents(text)
        if df_intent is not None:
            parsed_text = df_intent
            returned_item = random.choice(self.dialogue_flow_map[parsed_text])
        else:
            flag, idx, idx_list = self.check_prompt_availability(text)

            if flag:
                parsed_text = self.parsed_texts[idx]
                for (_, temp_str) in idx_list:
                    try:
                        _ = int(temp_str)
                        temp = parsed_text.split()
                        for i, item in enumerate(temp):
                            try:
                                _id = int(item)
                                temp[i] = str(temp_str)
                                break
                            except ValueError:
                                pass
                        parsed_text = " ".join(temp)

                    except ValueError:
                        pass
                parse_tree = None
            else:
                parse_tree, parsed_text = self.compute_parse_text(text)
                # parsed_text = "filter id 213 and nlpattribute all [E]"

            app.logger.info(f"parsed text: {parsed_text}")

            # Run the action in the conversation corresponding to the formal grammar
            returned_item = run_action(
                user_session_conversation, parse_tree, parsed_text)

        self.parsed_text = parsed_text

        username = user_session_conversation.username

        response_id = self.gen_almost_surely_unique_id()
        logging_info = self.build_logging_info(self.bot_name,
                                               username,
                                               response_id,
                                               text,
                                               parsed_text,
                                               returned_item)
        # add the parsed operation
        self.conversation.previous_operations.append(self.parsed_text)
        if self.suggestions:  # and random.random() > 0.5:
            self.suggested_operation, suggestion_text = self.pick_relevant_operation(parsed_text)
            returned_item += suggestion_text
        self.log(logging_info)

        # Concatenate final response, parse, and conversation representation
        # This is done so that we can split both the parse and final
        # response, then present all the data
        final_result = returned_item + f"<>{response_id}"

        return final_result
