from transformers import AutoTokenizer, BertTokenizer


class HFTokenizer:
    def __init__(self, model_id, mode='auto'):
        self.model_id = model_id
        self.tokenizer = None
        self.mode = mode
        self.create_tokenizer()

    def create_tokenizer(self):
        if self.mode == "auto":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        elif self.mode == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.model_id)
