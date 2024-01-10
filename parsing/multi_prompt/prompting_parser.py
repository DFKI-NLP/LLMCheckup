import sys
from os.path import dirname, abspath

parent = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(parent)

from parsing.multi_prompt.defined_prompts import operation_type_prompt, nlpattribute_prompt, rationalize_prompt, \
    show_prompt, keywords_prompt, similar_prompt, augment_prompt, cfe_prompt, mistake_prompt, predict_prompt, \
    score_prompt, operations_wo_attributes, operation2prompt, tutorial_operations, tutorial2operation, \
    valid_operation_names, valid_operation_prompt_samples, valid_operation_meanings, operation2attributes, \
    operation2tutorial, operation_needs_id

from sentence_transformers import SentenceTransformer, util
from word2number import w2n

from transformers import GenerationConfig, AutoModelForCausalLM, GPTQConfig, AutoTokenizer


class MultiPromptParser:
    def __init__(self, decoder_model, tokenizer, st_model, device):
        self.decoder_model = decoder_model
        self.tokenizer = tokenizer
        self.st_model = st_model
        self.device = device
        self.encoded_operations_st = self.st_model.encode(valid_operation_names, convert_to_tensor=True)
        self.encoded_op_meanings_st = self.st_model.encode(valid_operation_meanings, convert_to_tensor=True)
        self.attribute2st = dict()
        for operation in operation2attributes:
            for attribute in operation2attributes[operation]:
                self.attribute2st[attribute] = self.st_model.encode(attribute, convert_to_tensor=True)
        if "llama" in self.decoder_model.name_or_path.lower():
            max_new_tokens = 20
        else:
            max_new_tokens = 10
        self.generation_config = GenerationConfig(
            penalty_alpha=0.6,
            do_sample = True,
            top_k=5,
            top_p=0.95,
            temperature=0.1,
            repetition_penalty=1.2,
            max_new_tokens=max_new_tokens,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=2
        )


    def in_vocabulary(self, operation):
        operation_words = operation.split()
        for op in valid_operation_names:
            if op in operation_words:
                return True
        return False

    """ checks if the parsed attributes are valid tokens for the operation """

    def check_attribute(self, attribute, main_operation, user_input):
        attribute = attribute.strip()
        for punct in [".", ",", ":"]:
            attribute = attribute.replace(punct, "")
        valid_attributes = []
        if main_operation in operation2attributes:
            valid_attributes += operation2attributes[main_operation]
        # number can always be a valid attribute and needs to be checked first
        try:
            number = w2n.word_to_num(attribute)
            number = str(number)
            if number in user_input:
                return number
        except:
            pass
        if attribute in valid_attributes:
            return attribute
        else:
            # check the embedding similarity to the valid attributes
            if (len(valid_attributes) > 0):
                valid_attributes_st = self.st_model.encode(valid_attributes, convert_to_tensor=True)
                if attribute in self.attribute2st:
                    attribute_st = self.attribute2st[attribute]
                else:
                    attribute_st = self.st_model.encode(attribute, convert_to_tensor=True)
                attr_scores = util.cos_sim(attribute_st, valid_attributes_st)[0].tolist()
                if max(attr_scores) >= 0.5:
                    return valid_attributes[attr_scores.index(max(attr_scores))]
                else:
                    return ""
            else:
                return ""

    """generates the parse given a user input and a prompt"""

    def generate_with_prompt(self, prompt, user_input):
        inputs = self.tokenizer(prompt + "\nInput: " + user_input + " Output:", return_tensors="pt").to(self.device)

        outputs = self.decoder_model.generate(**inputs, generation_config=self.generation_config)

        parsed_operation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        offset = len("Output:")
        parsed_operation = parsed_operation[parsed_operation.rindex("Output:") + offset:].strip()

        # for falcon and pythia only since they are not good at parsing ids
        # we check if we have id word in the user input and in case
        # "filter id" is missing in the parsed output we add it to the parse
        if "falcon" in self.decoder_model.name_or_path.lower() or "pythia" in self.decoder_model.name_or_path.lower():
            op_id = self.find_id(user_input).strip()
            if len(op_id) > 0:
                parsed_operation = "filter id " + op_id + " and " + parsed_operation

        # each operation consists of:
        # - prefix ("filter id ...")
        # - main_operation (e.g., "cfe", "similar")
        # - attributes (e.g., "topk 5")
        op_prefix = ""
        main_operation = parsed_operation
        attributes = ""
        splitted_op = parsed_operation.split()
        if parsed_operation.startswith("filter id") and len(splitted_op) > 4:
            op_prefix = " ".join(splitted_op[:4]) + " and "
            main_operation = " ".join(splitted_op[4:])

        splitted_main_op = main_operation.split()
        if len(splitted_main_op) > 0:
            main_operation = splitted_main_op[0]

        if not (self.in_vocabulary(main_operation)):
            # find a similar operation with SBERT
            main_operation = self.find_similar_operation(user_input)

        # check that the attributes are valid
        if len(splitted_main_op) > 1:
            attributes = " ".join(splitted_main_op[1:])
            if "[E]" in attributes:
                attributes = attributes[:attributes.rindex("[E]") + 3]
            checked_attributes = []
            for attribute in attributes.split():
                attribute = self.check_attribute(attribute, main_operation, user_input)
                if not (attribute in checked_attributes):
                    if len(checked_attributes) <= 2 and len(attribute) > 0:
                        checked_attributes.append(attribute)
                    else:
                        break
            if len(checked_attributes) == 0 and (main_operation in operation2attributes) and len(operation2attributes[
                                                                                                     main_operation]) > 0:  # some operations can have numbers but no other fixed attributes
                attributes = operation2attributes[main_operation][0]
            else:
                attributes = " ".join(checked_attributes)

        parsed_operation = op_prefix + main_operation
        if len(attributes) > 0:
            parsed_operation += " " + attributes

        return parsed_operation

    """fallback in case the id was not parsed correctly"""

    def find_id(self, user_input):
        user_input = user_input.lower()
        correct_id = ""
        id_words = ["id", "instance", "sample"]
        for id_word in id_words:
            if id_word in user_input:
                # if there is no id, e.g.: "counterfactual for this id"
                for id_candidate in user_input[user_input.rindex(id_word) + len(id_word):].split():
                    correct_id = id_candidate.replace("?", "").replace(".", "").replace(",", "")
                    try:
                        correct_id = w2n.word_to_num(correct_id)
                        correct_id = str(correct_id)
                        return correct_id
                    except:
                        correct_id = ""
        return correct_id

    """implements similarity check with SBERT between the user input and available operations"""

    def find_similar_operation(self, user_input):
        encoded_user_input = self.st_model.encode(user_input, convert_to_tensor=True)
        op_scores = util.cos_sim(encoded_user_input, self.encoded_op_meanings_st)[0].tolist()
        best_match_op = valid_operation_names[op_scores.index(max(op_scores))]
        return best_match_op

    """performs multi-prompt parsing for operations and their attributes"""

    def parse_user_input(self, user_input):
        parsed_operation = self.generate_with_prompt(operation_type_prompt, user_input).replace("[E]", "").strip()
        op_start = parsed_operation.split()[0]
        if not ((op_start in valid_operation_names) or (op_start == "filter")):
            parsed_operation = self.find_similar_operation(user_input)
        if parsed_operation in operations_wo_attributes:
            parsed_operation = parsed_operation + " [E]"
        else:
            if parsed_operation in operation2prompt.keys():
                operation_prompt = operation2prompt[parsed_operation]
                parsed_operation = self.generate_with_prompt(operation_prompt, user_input)
        # if we have an id we cannot have a tutorial operation
        if parsed_operation in tutorial2operation:
            user_input_words = user_input.split()
            for word in user_input_words:
                if word in ["id", "sample", "instance", "this", "it"]:
                    no_tutorial = True
                    operation = tutorial2operation[parsed_operation]
                    operation_prompt = operation2prompt[operation]
                    parsed_operation = self.generate_with_prompt(operation_prompt, user_input)
                    break
        non_repeated_tokens = set()
        splitted_parsed_op = parsed_operation.split()
        filtered_pased_op_tokens = []
        # check that tokens other than numbers do not appear more than once in the parsed output
        for token in splitted_parsed_op:
            is_number_token = isinstance(token, int)
            if (not(is_number_token) and not(token in non_repeated_tokens)) or is_number_token:
                filtered_pased_op_tokens.append(token)
                non_repeated_tokens.add(token)
            if token == "[E]":
                break

        for i, token in enumerate(filtered_pased_op_tokens):
            if token == "id" and not (filtered_pased_op_tokens[i + 1] in user_input):
                # replace hallucinated id
                found_id = self.find_id(user_input)
                if len(found_id) > 0:
                    filtered_pased_op_tokens[i + 1] = found_id
                else:
                    filtered_pased_op_tokens = filtered_pased_op_tokens[i + 3:]
                break
        parsed_operation = " ".join(filtered_pased_op_tokens).strip()

        # check if we parsed a standard operation but there is no id
        splitted_op = parsed_operation.split()
        if not (parsed_operation.startswith("filter")):
            main_operation = splitted_op[0]
        elif len(splitted_op) > 4:
            main_operation = splitted_op[4]

        #if not "filter id " in parsed_operation and main_operation in operation2tutorial:  # no filter for tutorial operations
        #    parsed_operation = operation2tutorial[main_operation]
        if "filter id " in parsed_operation and main_operation in tutorial2operation:  # if we have filter it's not the tutorial
            main_operation = tutorial2operation[main_operation]
            splitted_op = parsed_operation.split()
            if len(splitted_op) > 3:
                id_token = splitted_op[2]
            parsed_operation = "filter id " + id_token + " and " + main_operation
        elif not "filter id " in parsed_operation and main_operation in operation_needs_id:
            # check if we have a number in the parsed text, use it as id
            for token in parsed_operation.split():
                for punct in [".", ",", ":"]:
                    token = token.replace(punct, "")
                try:
                    id_num = w2n.word_to_num(token)
                    if str(id_num) in user_input:
                        parsed_operation = "filter id " + token + " and " + main_operation
                        break
                except:
                    continue

        if not (parsed_operation.endswith(" [E]")):
            parsed_operation += " [E]"
        # show by defult 10 top attributed tokens if the number was not identified
        if (parsed_operation.endswith("topk [E]")):
            parsed_operation = parsed_operation.replace("topk [E]", "topk 10 [E]")
        # remove invalid insertion of the filter (e.g., happens with the inputs like "counterfactual for this id")
        parsed_operation = parsed_operation.replace("filter id and", "")
        if parsed_operation.split()[0] in tutorial_operations:
            parsed_operation = "qatutorial " + parsed_operation

        return parsed_operation


### code execution ###

if __name__ == "__main__":
    # parsing accuracy evaluation (exact matches)

    model_id = "TheBloke/Llama-2-7b-Chat-GPTQ" #"EleutherAI/pythia-2.8b-v0" #"TheBloke/Llama-2-7b-Chat-GPTQ"  #"TheBloke/Mistral-7B-v0.1-GPTQ" #"tiiuae/falcon-rw-1b"

    # loading model and tokenizer
    if "GPTQ" in model_id:
        quantization_config = GPTQConfig(bits=4, disable_exllama=True)
        decoder_model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, device_map="auto", quantization_config=quantization_config)
    else:
        decoder_model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, device_map="auto")

    decoder_model.config.pad_token_id = decoder_model.config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # loading SBERT
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    device = "cuda"
    # initializing the multi-prompt parsing model
    parser = MultiPromptParser(decoder_model, tokenizer, st_model, device)
    evaluation_file = "experiments/testset_without_flag.txt"  # "experiments/testset.txt" # "experiments/testset_without_flag.txt"
    correct_parses = []
    user_inputs = []
    with open(evaluation_file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.endswith("[E]"):
                correct_parses.append(line)
            elif len(line) > 0:
                user_inputs.append(line)
    sys_parses = []
    for user_input in user_inputs:
        parsed_operation = parser.parse_user_input(user_input)
        print(parsed_operation + " >>> " + user_input)
        sys_parses.append(parsed_operation)
    matched = 0
    assert (len(sys_parses) == len(correct_parses))
    for i in range(len(sys_parses)):
        if sys_parses[i] == correct_parses[i]:
            matched += 1
    print("matched:", matched, "total:", len(correct_parses), "acc:", matched / len(correct_parses))
