"""Parsing engine that converts natural language to the grammar.

This class 'decodes' natural language inputs into the grammar. There are several parsing options
supported, such as fine-tuned t5 models, few-shot gpt-j models, and KNN.
"""
import gin
from transformers import AutoModelForCausalLM, AutoTokenizer
from petals import AutoDistributedModelForCausalLM
import torch


@gin.configurable
class Decoder:
    """Class that defines parser options."""

    def __init__(self,
                 parsing_model_name: str,
                 in_8_bits: bool,
                 conversation_config_file: str = None,
                 no_init: bool = False,
                 use_guided_decoding: bool = True,
                 dataset_name: str = None):
        """Init

        Arguments:
            parsing_model_name: The name of the parsing model. The currently supported are:
                                   * t5 models: if using a t5 model, parsing_model_name must be the name of
                                     the model path or huggingface directory
                                   * gpt-j few shot models: if using few-shot, this must be the name of the
                                     hugging face directory, e.g., 'EleutherAI/gpt-j-6B'
                                   * nearest-neighbor: if this, will use knn on the prompts to parse
                                Note, that for t5 and gpt models to be handled correctly, 't5' or 'gpt'
                                **must** be specified in parsing_model_name.
            conversation_config_file: The gin config location for the conversation.
            no_init: If True, will not init any parsing model
            use_guided_decoding: Whether to use guided decoding
            dataset_name: The name of the dataset
        """
        self.gen_completions = None
        self.use_guided_dec = use_guided_decoding
        self.gpt_parser_initialized = False
        self.gpt_model = None
        self.gpt_tokenizer = None
        self.parser_name = parsing_model_name
        self.init_model(parsing_model_name,
                        in_8_bits,
                        no_init=no_init,
                        dataset_name=dataset_name,
                        conversation_config_file=conversation_config_file)

    def init_model(self,
                   parsing_model_name: str,
                   in_8_bits,
                   no_init: bool = False,
                   dataset_name: str = None,
                   conversation_config_file: str = None):
        """Initializes the model

        Args:
            conversation_config_file: The conversation gin config file path
            dataset_name: The semantic name of the dataset
            no_init: Do not init the model
            parsing_model_name: the name of the model
            config_file: a gin config file required for t5 models
        """

        # Does not initialize a model
        if no_init:
            return

        if "petals-team" in parsing_model_name:
            """p2p models from petals-team"""
            self.gpt_tokenizer = AutoTokenizer.from_pretrained(parsing_model_name)
            self.gpt_model = AutoDistributedModelForCausalLM.from_pretrained(parsing_model_name)
            self.gpt_parser_initialized = True

            from parsing.gpt.few_shot_inference import get_few_shot_predict_f
            predict_f = get_few_shot_predict_f(model=self.gpt_model,
                                               tokenizer=self.gpt_tokenizer,
                                               use_guided_decoding=self.use_guided_dec)

            def complete(prompt, grammar):
                return predict_f(text=prompt, grammar=grammar)
        elif "GPTQ" in parsing_model_name:
            """GPTQ quantized model"""
            self.gpt_tokenizer = AutoTokenizer.from_pretrained(parsing_model_name)
            self.gpt_model = AutoModelForCausalLM.from_pretrained(parsing_model_name, torch_dtype=torch.float16,
                                                                  device_map="auto")
            self.gpt_model.config.pad_token_id = self.gpt_model.config.eos_token_id
            self.gpt_parser_initialized = True

            from parsing.gpt.few_shot_inference import get_few_shot_predict_f
            predict_f = get_few_shot_predict_f(model=self.gpt_model,
                                               tokenizer=self.gpt_tokenizer,
                                               use_guided_decoding=self.use_guided_dec)

            def complete(prompt, grammar):
                return predict_f(text=prompt, grammar=grammar)
        elif "Llama" in parsing_model_name or "Mistral" in parsing_model_name:
            """original model"""
            if not self.gpt_parser_initialized:
                self.gpt_tokenizer = AutoTokenizer.from_pretrained(parsing_model_name)
                if in_8_bits:
                    self.gpt_model = AutoModelForCausalLM.from_pretrained(parsing_model_name, device_map='cpu',
                                                                          load_in_8bit=True)
                else:
                    self.gpt_model = AutoModelForCausalLM.from_pretrained(parsing_model_name)
                self.gpt_model.config.pad_token_id = self.gpt_model.config.eos_token_id
                self.gpt_parser_initialized = True

            from parsing.gpt.few_shot_inference import get_few_shot_predict_f
            predict_f = get_few_shot_predict_f(model=self.gpt_model,
                                               tokenizer=self.gpt_tokenizer,
                                               use_guided_decoding=self.use_guided_dec)

            def complete(prompt, grammar):
                return predict_f(text=prompt, grammar=grammar)
        elif parsing_model_name == "nearest-neighbor":
            def complete(prompt, _):
                split_prompts = prompt.split("\n")
                # prompt should be third back, considering
                # "parsed:" is last and the user input is 2nd last
                # then line break
                last_prompt = split_prompts[-4][len("parsed: "):]
                responses = {"generation": "\n".join(split_prompts) + last_prompt}
                return responses
        else:
            raise NotImplementedError

        self.gen_completions = complete

    def complete(self, prompt: str, grammar: str = None):
        """Run a completion."""
        assert self.gen_completions is not None, "Must run init_model first!"
        completed = self.gen_completions(prompt, grammar)
        return completed
