from datasets import load_dataset
from typing import Dict


def get_dataset_index_range(dataset, dataset_config):
    start = 0 if 'start' not in dataset_config else dataset_config['start']
    if start < 0:
        start = 0

    end = len(dataset) if 'end' not in dataset_config else dataset_config['end']
    if end < 0:
        end = len(dataset)

    return range(start, end)


def get_dataset(tokenizer, n=None):
    if n is None:
        dataset = load_dataset("super_glue", "boolq", split="validation")
    else:
        dataset = load_dataset("super_glue", "boolq", split="validation[0:" + str(n) + "]")
    dataset_config = {
        "text_field": ["question", "passage"],
        "columns": ['input_ids', 'attention_mask', 'special_tokens_mask', 'labels'],
        "batch_size": 1,
    }
    tokenization_config = {
        "max_length": 512,
        "padding": "max_length",
        "return_tensors": "np",
        "truncation": True,
        "special_tokens_mask": True,
    }

    def encode(instances):
        return tokenizer(instances[text_field],
                         truncation=tokenization_config['truncation'],
                         padding=tokenization_config['padding'],
                         max_length=tokenization_config['max_length'],
                         return_special_tokens_mask=tokenization_config['special_tokens_mask'])

    def encode_pair(instances):
        return tokenizer(instances[text_field[0]],
                         instances[text_field[1]],
                         truncation=tokenization_config['truncation'],
                         padding=tokenization_config['padding'],
                         max_length=tokenization_config['max_length'],
                         return_special_tokens_mask=tokenization_config['special_tokens_mask'])

    dataset = dataset.select(indices=get_dataset_index_range(dataset, dataset_config))

    if 'text_field' in dataset_config:
        text_field = dataset_config['text_field']
    else:
        text_field = 'text'
    encode_fn = encode_pair if type(text_field) == list and len(text_field) else encode
    dataset = dataset.map(encode_fn, batched=True, batch_size=dataset_config['batch_size'])

    if 'label_field' in dataset_config:
        label_field = dataset_config['label_field']
    else:
        label_field = 'label'

    def get_label(example: Dict):
        return example[label_field]

    dataset = dataset.map(lambda examples: {'labels': get_label(examples)},
                          batch_size=dataset_config['batch_size'])  # batched=True,
    dataset.set_format(type='torch', columns=dataset_config['columns'])
    return dataset
