import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from datasets import Dataset

from logic.utils import get_user_questions_and_parsed_texts

model_id = "meta-llama/Llama-2-7b-chat-hf"
# model_id = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj",
                    "up_proj",
                    "o_proj",
                    "k_proj",
                    "down_proj",
                    "gate_proj",
                    "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

user_question, parsed_text = get_user_questions_and_parsed_texts()

ds = Dataset.from_dict({"user_text": user_question,
                        "parsed_text": parsed_text})
ds = ds.map(lambda samples: tokenizer(samples["user_text"]), batched=True)

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=ds,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=15,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=1,
        output_dir="../outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
wandb.login(key="1f490f5a176c4a3832c822f7cd72cab56c1f5d3a")
trainer.train()
trainer.save_model(f"../outputs/{model_id.split('/')[1]}")
