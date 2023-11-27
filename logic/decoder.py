"""Parsing engine that converts natural language to the grammar.

This class 'decodes' natural language inputs into the grammar.
"""
import gin
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, GPTNeoXForCausalLM
from petals import AutoDistributedModelForCausalLM


@gin.configurable
class Decoder:
    """Class that defines parser options."""

    def __init__(self,
                 parsing_model_name: str,
                 load_in_4bits: bool = False,
                 no_init: bool = False,
                 use_guided_decoding: bool = True,
                 dataset_name: str = None):
        """Init

        Arguments:
            parsing_model_name: The name of the parsing model.
            load_in_4bits: if True, then load model in 4 bits
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
                        load_in_4bits,
                        no_init=no_init)

    def init_model(self,
                   parsing_model_name: str,
                   load_in_4bits: bool,
                   no_init: bool = False):
        """Initializes the model

        Args:
            :param no_init: Do not init the model
            :param parsing_model_name: the name of the model
            :param load_in_4bits: load model in 8 bits
        """

        # Does not initialize a model
        if no_init:
            return

        if "petals-team" in parsing_model_name:
            """p2p models from petals-team"""
            self.gpt_tokenizer = AutoTokenizer.from_pretrained(parsing_model_name, device_map="auto")
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
            quantization_config = GPTQConfig(bits=4, tokenizer=self.gpt_tokenizer)
            self.gpt_model = AutoModelForCausalLM.from_pretrained(parsing_model_name,
                                                                  quantization_config=quantization_config,
                                                                  device_map="auto")
            self.gpt_model.config.pad_token_id = self.gpt_model.config.eos_token_id
            self.gpt_parser_initialized = True

            from parsing.gpt.few_shot_inference import get_few_shot_predict_f
            predict_f = get_few_shot_predict_f(model=self.gpt_model,
                                               tokenizer=self.gpt_tokenizer,
                                               use_guided_decoding=self.use_guided_dec)

            def complete(prompt, grammar):
                return predict_f(text=prompt, grammar=grammar)
        elif "Llama" in parsing_model_name or "Mistral" in parsing_model_name or "falcon" in parsing_model_name:
            """original model"""
            if not self.gpt_parser_initialized:
                self.gpt_tokenizer = AutoTokenizer.from_pretrained(parsing_model_name)
                if load_in_4bits:
                    self.gpt_model = AutoModelForCausalLM.from_pretrained(parsing_model_name, device_map='cuda:0',
                                                                          load_in_4bit=load_in_4bits)
                else:
                    self.gpt_model = AutoModelForCausalLM.from_pretrained(parsing_model_name, device_map="auto")
                self.gpt_model.config.pad_token_id = self.gpt_model.config.eos_token_id
                self.gpt_parser_initialized = True

            from parsing.gpt.few_shot_inference import get_few_shot_predict_f
            predict_f = get_few_shot_predict_f(model=self.gpt_model,
                                               tokenizer=self.gpt_tokenizer,
                                               use_guided_decoding=self.use_guided_dec)

            def complete(prompt, grammar):
                return predict_f(text=prompt, grammar=grammar)
        elif "pythia" in parsing_model_name:
            if not self.gpt_parser_initialized:
                self.gpt_tokenizer = AutoTokenizer.from_pretrained(parsing_model_name)

                if load_in_4bits:
                    self.gpt_model = GPTNeoXForCausalLM.from_pretrained(parsing_model_name, device_map='cuda:0',
                                                                        load_in_8bit=load_in_4bits)
                else:
                    self.gpt_model = GPTNeoXForCausalLM.from_pretrained(parsing_model_name, device_map="auto")
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
