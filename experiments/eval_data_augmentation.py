import json
import random

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import nlpaug.augmenter.word as naw
from transformers import AutoTokenizer, AutoModelForCausalLM

from actions.prediction.predict import get_demonstrations, convert_str_to_options
from actions.prediction.predict_grammar import COVID_GRAMMAR, ECQA_GRAMMAR
from parsing.guided_decoding.gd_logits_processor import GuidedParser, GuidedDecodingLogitsProcessor


def get_prediction(tokenizer, model, idx, first, second, ds, num_shot=3):
    selected_first_field, selected_second_field, labels = get_demonstrations(idx, num_shot, ds)

    if ds == "covid_fact":

        prompt_template = "Each 3 items in the following list contains the claims, evidence and prediction. Your task " \
                          "is to predict the claims based on evidence as one of the labels: REFUTED, SUPPORTED.\n"

        for i in range(num_shot):
            prompt_template += f"claim: {selected_first_field[i]}\n"
            prompt_template += f"evidence: {selected_second_field[i]}\n"
            prompt_template += f"prediction: {int2label[labels[i]]}\n"
            prompt_template += "\n"

        prompt_template += f"claim: {first}\n"
        prompt_template += f"evidence: {second}\n"
        prompt_template += f"prediction: "

        parser = GuidedParser(COVID_GRAMMAR, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1])
    else:
        prompt_template = "Each 3 items in the following list contains the question, choice and prediction. Your task " \
                          "is to choose one of the choices as the answer for the question\n"
        for i in range(num_shot):
            prompt_template += f"question: {selected_first_field[i]}\n"
            prompt_template += f"choices: {convert_str_to_options(selected_second_field[i])}\n"
            prompt_template += f"prediction: {labels[i] + 1}\n"
            prompt_template += "\n"

        prompt_template += f"question: {first}\n"
        prompt_template += f"choices: {convert_str_to_options(second)}\n"
        prompt_template += f"prediction: "

        parser = GuidedParser(ECQA_GRAMMAR, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1])

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.to(model.device.type)

    guided_preprocessor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])

    with torch.no_grad():
        generation = model.greedy_search(input_ids, logits_processor=guided_preprocessor,
                                         pad_token_id=model.config.pad_token_id,
                                         eos_token_id=parser.eos_token, device=model.device.type)
    try:
        prediction = tokenizer.decode(generation[0]).split(prompt_template)[1].split(" [e]")[0].split(" ")[1]
    except IndexError:
        if ds == "ecqa":
            prediction = tokenizer.decode(generation[0]).split(prompt_template)[0][-5]
        else:
            temp = tokenizer.decode(generation[0]).split(prompt_template)[0][-11:-4]
            if temp == "REFUTED":
                prediction = temp
            else:
                prediction = "SUPPORTED"
    # prediction = tokenizer.decode(generation[0]).split(prompt_template)[1].split(" ")[2].split("<s>")[0]

    return prediction


if __name__ == "__main__":
    ds = "covid_fact"
    # ds = "ecqa"
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    # model_name = "mistralai/Mistral-7B-v0.1"
    model_name = "EleutherAI/pythia-2.8b-v0"
    # model_name = "tiiuae/falcon-rw-1b"

    similarity_model = SentenceTransformer("all-mpnet-base-v2")

    if ds == "covid_fact":
        df = pd.read_csv("../data/COVIDFACT_dataset.csv")
        claims = list(df["claims"])
        evidences = list(df["evidences"])

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:0', load_in_8bit=True)

        model.config.pad_token_id = model.config.eos_token_id

        random_list = random.sample(range(0, len(claims)), 100)

        int2label = {1: "REFUTED", 0: "SUPPORTED"}

        aug = naw.SynonymAug(aug_src='wordnet')

        json_list = []

        for idx in tqdm(random_list):

            pre_prediction = get_prediction(tokenizer, model, idx, claims[idx], evidences[idx], ds)
            augmented_first_field = aug.augment(claims[idx])
            augmented_second_field = aug.augment(evidences[idx])
            post_prediction = get_prediction(tokenizer, model, idx, augmented_first_field, augmented_second_field, ds)

            query_embedding = similarity_model.encode(claims[idx])
            sent_embeddings = similarity_model.encode(augmented_first_field)
            claim_cos_sim = round(util.cos_sim(query_embedding, sent_embeddings)[0].tolist()[0], 2)

            query_embedding = similarity_model.encode(evidences[idx])
            sent_embeddings = similarity_model.encode(augmented_second_field)
            evidence_cos_sim = round(util.cos_sim(query_embedding, sent_embeddings)[0].tolist()[0], 2)

            agreement = 1 if (pre_prediction == post_prediction) else 0

            print(pre_prediction, post_prediction)

            json_list.append({
                "idx": idx,
                "agreement": agreement,
                "claim_cos_sim": claim_cos_sim,
                "evidence_cos_sim": evidence_cos_sim
            })

    else:
        df = pd.read_csv("../data/ECQA_dataset.csv")
        texts = list(df["texts"])
        choices = list(df["choices"])

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:0', load_in_8bit=True)

        model.config.pad_token_id = model.config.eos_token_id

        random_list = random.sample(range(0, len(texts)), 100)

        aug = naw.SynonymAug(aug_src='wordnet')

        json_list = []

        for idx in tqdm(random_list):
            pre_prediction = get_prediction(tokenizer, model, idx, texts[idx], choices[idx], ds=ds)
            augmented_first_field = aug.augment(texts[idx])
            # augmented_second_field = aug.augment(convert_str_to_options(choices[idx]))
            post_prediction = get_prediction(tokenizer, model, idx, augmented_first_field, choices[idx], ds)

            query_embedding = similarity_model.encode(texts[idx])
            sent_embeddings = similarity_model.encode(augmented_first_field)
            claim_cos_sim = round(util.cos_sim(query_embedding, sent_embeddings)[0].tolist()[0], 2)

            # query_embedding = similarity_model.encode(convert_str_to_options(choices[idx]))
            # sent_embeddings = similarity_model.encode(augmented_second_field)
            # evidence_cos_sim = round(util.cos_sim(query_embedding, sent_embeddings)[0].tolist()[0], 2)

            agreement = 1 if (pre_prediction == post_prediction) else 0
            print(f"idx: {idx}, pre: {pre_prediction}, post: {post_prediction}")

            json_list.append({
                "idx": idx,
                "agreement": agreement,
                "claim_cos_sim": claim_cos_sim,
                # "evidence_cos_sim": evidence_cos_sim
            })

    jsonString = json.dumps(json_list)
    jsonFile = open(f"../cache/{ds}/{ds}_data_augmentation_{model_name.split('/')[1]}.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()