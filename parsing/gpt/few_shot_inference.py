"""Few shot inference via gpt-j / neo series models."""
import sys
from os.path import dirname, abspath

import gin
import torch
from transformers import MaxLengthCriteria

parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from parsing.guided_decoding.gd_logits_processor import GuidedParser, GuidedDecodingLogitsProcessor


@gin.configurable
def get_few_shot_predict_f(
        model,
        tokenizer,
        device: str = "cpu",
        use_guided_decoding: bool = True
    ):
    """Gets the few shot prediction model.

    Args:
        model: the gpt series model for few shot prediction
        tokenizer: the gpt tokenizer for the chosen model
        device:
        use_guided_decoding: whether to use guided decoding
    """

    def predict_f(text: str, grammar: str):
        """The function to get guided decoding."""
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device.type)

        if use_guided_decoding:
            parser = GuidedParser(grammar, tokenizer, model="gpt")
            guided_preprocessor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])
            with torch.no_grad():
                generation = model.greedy_search(input_ids,
                                                 logits_processor=guided_preprocessor,
                                                 eos_token_id=parser.eos_token,
                                                 pad_token_id=model.config.pad_token_id,
                                                 device=model.device.type)
        else:
            stopping_criteria = MaxLengthCriteria(max_length=200)
            generation = model.greedy_search(input_ids,
                                             stopping_criteria=stopping_criteria,
                                             device=model.device.type)

        decoded_generation = tokenizer.decode(generation[0])
        return {"generation": decoded_generation}

    return predict_f
