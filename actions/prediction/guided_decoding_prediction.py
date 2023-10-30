import json
import os
import copy
import re
import random

import pandas as pd
from tqdm import tqdm

from lark import Lark
import numpy as np

import torch

from transformers import LogitsProcessor, AutoModelForCausalLM, AutoTokenizer


class GuidedDecodingLogitsProcessor(LogitsProcessor):
    def __init__(self, parser, prompt_length, filter_value=-float("Inf"), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = parser
        self.prompt_length = prompt_length
        self.filter_value = filter_value

    def __call__(self, input_ids, scores):
        valid_tokens = torch.ones_like(scores) * self.filter_value

        # The tokens generated so far
        for b in range(scores.shape[0]):
            generated_tokens = input_ids[b, self.prompt_length:].cpu().tolist()
            next_tokens = self.parser.next_tokens(generated_tokens)
            int_next_tokens = np.array([int(t) for t in next_tokens])

            # Adjust the scores to allow only valid tokens
            valid_tokens[b, int_next_tokens] = scores[b, int_next_tokens]
        return valid_tokens


class GuidedParser:
    """A class defining the mapping between text grammar and tokenized grammar."""
    def __init__(self, init_grammar, tokenizer, model, eos_token=None):

        # The grammar with natural language text
        self.text_grammar = init_grammar

        # The natural language parser
        self.text_parser = Lark(self.text_grammar, parser="lalr")

        # The hugging face tokenizer
        self.tokenizer = tokenizer

        # Store the model being used. This influences some decoding settings
        self.model = model

        # The grammar compiled with tokens from the hugging face tokenizer
        self.token_grammar = self._compile_grammar(self.text_grammar, self.tokenizer)

        # The tokenized parser
        self.token_parser = Lark(self.token_grammar, parser="lalr")

        self.terminal_lookup = {}

        for terminal in self.token_parser.terminals:
            self.terminal_lookup[terminal.name] = terminal.pattern.value

        if eos_token is None:
            if model == "t5":
                self.eos_token = tokenizer.encode(" [e]")[-2]
            elif model == "gpt":
                self.eos_token = tokenizer.encode(" [e]")[-1]
            else:
                raise NameError(f"don't know model {model}")
        else:
            self.eos_token = eos_token

    def _compile_grammar(self, grammar, tokenizer):
        """Compiles a grammar into tokens."""

        # Create the tokenizer grammar
        tokenized_grammar = copy.deepcopy(grammar)

        # Find all the terminals
        terminals = re.findall('"([^"]*)"', grammar)

        # Store existing terminals
        existing_terms = {}

        # Records the update rules for the terminals
        indx = 0
        for term in tqdm(terminals):
            tokens = tokenizer.encode(term)

            replacement_rule = "("
            for tok in tokens:
                if tok == 1 and self.model == "t5":
                    continue
                # If it already exists, we don't want to add
                # the terminal again, just use the old one
                if tok in existing_terms:
                    name = existing_terms[tok]
                else:
                    name = f"ANON{indx} "
                    indx += 1
                    newrule = name + ": " + "\"" + str(tok) + "\""
                    tokenized_grammar += f"\n{newrule}"
                    existing_terms[tok] = name
                replacement_rule += name

            # Close the list of terminals
            replacement_rule += ")"

            # Update the terminal with the tokens
            tokenized_grammar = tokenized_grammar.replace("\"" + term + "\"",  replacement_rule)

        tokenized_grammar += "\n%ignore \" \""
        return tokenized_grammar

    def next_tokens(self, tokens):
        """Get the next tokens."""
        string_tokens = ' '.join([str(t) for t in tokens])
        interactive = self.token_parser.parse_interactive(string_tokens)
        interactive.exhaust_lexer()
        return [self.terminal_lookup[acc] for acc in interactive.accepts()]

# noqa: E501
GRAMMAR = r"""
?start: action
action: operation done 

done: " [e]"

operation: off | inoff 

inoff: " non-offensive"

off: " offensive"
"""
# df = pd.read_csv("../../data/offensive_val.csv")
df = pd.read_csv("/home/qwang/InterroLang/data/offensive_val.csv")
instances = list(df["text"])
# labels = list(df["label"])
id2str = {0: "non-offensive", 1: "offensive"}

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# model.to(device)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# few_shot = False
few_shot = True

if not few_shot:
    json_list = []
    for idx, instance in list(enumerate(instances)):
        input_ids = tokenizer(instance, return_tensors="pt").input_ids
        # input_ids = input_ids.to(device)

        parser = GuidedParser(GRAMMAR, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1])
        guided_preprocessor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])

        generation = model.greedy_search(input_ids, logits_processor=guided_preprocessor,
                                         pad_token_id=model.config.pad_token_id, eos_token_id=parser.eos_token)

        decoded_generation = tokenizer.decode(generation[0])
        try:
            prediction = decoded_generation.split(instance)[1].split("[e]")[0].strip()
        except:
            print(decoded_generation)
            temp = decoded_generation[-17:].split("[e]")[0].strip()
            if "non-offensive" in temp:
                prediction = "non-offensive"
            else:
                prediction = "offensive"
        # print({"generation": prediction})

        print({
            "idx": idx,
            "prediction": prediction
        })

        json_list.append({
            "idx": idx,
            "prediction": prediction
        })
    jsonString = json.dumps(json_list)
    jsonFile = open(f"/netscratch/qwang/guided/olid_gpt-neo-2.7b_zero_shot_prediction.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

else:
    json_list = []
    num_few_shots = [3, 5, 10, 15, 20]

    df = pd.read_csv("/home/qwang/InterroLang/data/offensive_train.csv")
    texts = list(df["text"])
    labels = list(df["label"])
    for num_few_shot in num_few_shots:
        for idx, instance in list(enumerate(instances)):
            prompt = ""
            for num in range(num_few_shot):
                rand_num = random.randint(0, len(instances) - 1)

                counter_0 = 0
                counter_1 = 0
                if counter_0 >= num_few_shot // 2:
                    while labels[rand_num] != 1:
                        rand_num = random.randint(0, len(instances) - 1)
                elif counter_1 >= num_few_shot // 2:
                    while labels[rand_num] != 0:
                        rand_num = random.randint(0, len(instances) - 1)

                prompt += f"Instance: {texts[rand_num]}\n"
                prompt += f"Parsed: {id2str[labels[rand_num]]} [e]\n"
            prompt += f"Instance: {instances[idx]}\n"
            prompt += "Parsed:"

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            # input_ids = input_ids.to(device)

            parser = GuidedParser(GRAMMAR, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1])
            guided_preprocessor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])

            generation = model.greedy_search(input_ids, logits_processor=guided_preprocessor,
                                             pad_token_id=model.config.pad_token_id, eos_token_id=parser.eos_token)

            decoded_generation = tokenizer.decode(generation[0])
            try:
                prediction = decoded_generation.split(prompt)[1].split("[e]")[0].strip()
            except:
                print(decoded_generation)
                temp = decoded_generation[-17:].split("[e]")[0].strip()
                if "non-offensive" in temp:
                    prediction = "non-offensive"
                else:
                    prediction = "offensive"

            print({
                "idx": idx,
                "prediction": prediction
            })

            json_list.append({
                "idx": idx,
                "prediction": prediction
            })
        jsonString = json.dumps(json_list)
        jsonFile = open(f"/netscratch/qwang/guided/olid_gpt-neo-2.7b_{num_few_shot}_shot_prediction.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()

