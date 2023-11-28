"""Computes the parsing accuracy of the model on a test suite."""
import argparse
import copy
import os
from os.path import dirname, abspath, join  # noqa: E402, F401
import sys

from datetime import datetime
from typing import Any

from pytz import timezone

import gin
import numpy as np
import pandas as pd
import torch

np.random.seed(0)
torch.manual_seed(0)
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from experiments.get_compositional_queries import run_iid_compositional_accuracies
from experiments.utils import load_test_data, check_correctness
from logic.core import ExplainBot  # noqa: E402, F401
# Needed for gin configs
from parsing.gpt.few_shot_inference import get_few_shot_predict_f  # noqa: E402, F401


def safe_name(model):
    return model.replace("/", "_")


def compute_accuracy(data, predict_func, verbose: bool = False, error_analysis: bool = False,
                     feature_names: Any = None):
    """Computes the parsing accuracy across some data."""
    nn_prompts = None
    misses, total = 0, 0
    parses_store = {}
    print(f"There are {len(data)} eval points", flush=True)

    for j, user_input in enumerate(data):
        has_all_parse_words = None
        if error_analysis:
            parse_text, nn_prompts = predict_func(user_input)
        else:
            parse_text = predict_func(user_input)

        if type(parse_text == tuple):
            if len(parse_text) == 2 and parse_text[1] is None:
                parse_text = parse_text[0]

        # Get the gold label parse
        correct_parse = data[user_input]

        # Do this to make sure extra spaces are ignored around input
        is_correct = check_correctness(parse_text, correct_parse)

        if error_analysis:
            parses = []
            print(nn_prompts)

            if nn_prompts is None:
                parses = []
            else:
                for prompt in nn_prompts:
                    split_prompt = prompt.split("parsed: ")
                    nn_parse = split_prompt[1]
                    parses.append(nn_parse)
            parses = " ".join(parses)

            has_all_parse_words = True
            for word in correct_parse.split(" "):

                # Cases not to look at parse word, i.e., things like numbers,
                # bools, and the feature names
                try:
                    float(word)
                    continue
                except:
                    pass
                if word == "true" or word == "false":
                    continue
                if word in feature_names:
                    continue

                if word not in parses:
                    has_all_parse_words = False

        if not is_correct:
            misses += 1

            if verbose:
                print(">>>>>")
                print(f"User: {user_input}")
                print(f"Parsed: {parse_text}")
                print(f"Correct: {correct_parse}")
                print(">>>>>")

        parses_store[f"parsed_text_{j}"] = parse_text
        parses_store[f"correct_parse_{j}"] = correct_parse
        parses_store[f"user_input_{j}"] = user_input

        if error_analysis:
            parses_store[f"includes_all_words_{j}"] = has_all_parse_words

        total += 1

        if j % 25 == 0:
            print(f"Error Rate | it {j} |  {round(misses / total, 3)}", flush=True)

    error_stat = misses / total

    if verbose:
        print(f"Final Error Rate: {round(error_stat, 3)}", flush=True)

    return error_stat, parses_store




def main():
    results = {
        "dataset": [],
        "model": [],
        "num_prompts": [],
        "accuracy": [],
        "in_domain_accuracy": [],
        "compositional_accuracy": [],
        "overall_accuracy": [],
        "total_in_domain": [],
        "total_compositional": [],
        "guided_decoding": [],
        "iid_errors_pct_not_all_words": [],
        "comp_errors_pct_not_all_words": []
    }
    model = args.model
    guided_decoding = args.gd
    dset = args.dataset

    program_only_text = ""
    if args.program_only:
        program_only_text += "-program-only"
    results_location = (f"./experiments/results_store/{safe_name(model)}_{dset}_gd-{guided_decoding}"
                        f"_debug-{args.debug}{program_only_text}.csv")
    if os.path.exists(results_location):
        print(f"Skipping already existing results file: f{results_location}\nPlease delete or move the results file to "
              f"produce new ones.")
        return

    print(f"-----------------", flush=True)
    print("Debug:", args.debug, flush=True)
    print("Dataset:", dset, flush=True)
    print("Model:", model, flush=True)

    config_dset_id = dset

    # test_suite = f"./experiments/testset.txt"
    test_suite = "./experiments/testset_without_flag.txt"

    if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
        if model == "nearest-neighbor":
            config = f"./configs/covid_fact_nn.gin"
        elif model == "meta-llama/Llama-2-7b-chat-hf":
            config = f"./configs/covid_fact_llama.gin"
        elif model == "mistralai/Mistral-7B-v0.1":
            config = f"./configs/covid_fact_mistral.gin"
        elif model == "tiiuae/falcon-rw-1b":
            config = f"./configs/covid_fact_falcon.gin"
        elif model == "petals":
            config = f"./configs/covid_fact_petals.gin"
        else:
            config = f"./configs/covid_fact_pythia.gin"
    elif sys.platform.startswith('win32') or sys.platform.startswith('cygwin'):
        if model == "\'nearest-neighbor\'":
            config = f"./configs/covid_fact_nn.gin"
        elif model == "\'meta-llama/Llama-2-7b-chat-hf\'":
            config = f"./configs/covid_fact_llama.gin"
        elif model == "\'mistralai/Mistral-7B-v0.1\'":
            config = f"./configs/covid_fact_mistral.gin"
        elif model == "\'tiiuae/falcon-rw-1b\'":
            config = f"./configs/covid_fact_falcon.gin"
        elif model == "\'petals\'":
            config = f"./configs/covid_fact_petals.gin"
        else:
            config = f"./configs/covid_fact_pythia.gin"
    else:
        raise OSError("Unknown operating system!")

    # Parse config
    gin.parse_config_file(config)
    testing_data = load_test_data(test_suite)

    # load the model
    bot, get_parse_text = load_model(dset, guided_decoding, model)

    error_analysis = False
    if "t5" not in model:
        error_analysis = True

    # load the number of prompts to perform in the sweep
    n_prompts_configs = load_n_prompts(model)

    if args.debug:
        n_prompts_configs = [10, 2]

    feature_names = copy.deepcopy(list(bot.conversation.stored_vars["dataset"].contents["X"].columns))

    for num_prompts in n_prompts_configs:

        # Set the bot to the number of prompts
        bot.set_num_prompts(num_prompts)
        print("Num prompts:", bot.prompts.num_prompt_template)
        assert bot.prompts.num_prompt_template == num_prompts, "Prompt update failing"

        # Compute the accuracy
        error_rate, all_parses = compute_accuracy(testing_data,
                                                  get_parse_text,
                                                  args.verbose,
                                                  error_analysis=error_analysis,
                                                  feature_names=feature_names)

        # Add parses to results
        for key in all_parses:
            if key not in results:
                results[key] = [all_parses[key]]
            else:
                results[key].append(all_parses[key])

        # Compute the compositional / iid accuracy splits
        iid_comp_results = run_iid_compositional_accuracies(dset,
                                                            all_parses,
                                                            bot,
                                                            program_only=args.program_only)

        in_acc, comp_acc, ov_all, total_in, total_comp, iid_pct_keys, comp_pct_keys = iid_comp_results

        # Store metrics
        results["total_in_domain"].append(total_in)
        results["total_compositional"].append(total_comp)
        results["in_domain_accuracy"].append(in_acc)
        results["compositional_accuracy"].append(comp_acc)
        results["overall_accuracy"].append(ov_all)
        results["guided_decoding"].append(guided_decoding)
        results["model"].append(model)
        results["dataset"].append(dset)
        results["accuracy"].append(1 - error_rate)
        results["num_prompts"].append(num_prompts)
        results["iid_errors_pct_not_all_words"].append(iid_pct_keys)
        results["comp_errors_pct_not_all_words"].append(comp_pct_keys)

        # Write everything to dataframe
        final_results = results
        result_df = pd.DataFrame(final_results)
        result_df.to_csv(results_location)
        print("Saved locally...", flush=True)

        # optionally upload to wandb
        if args.wandb:
            import wandb
            results_table = wandb.Table(data=result_df)
            if args.debug:
                table_name = "parsing-accuracy-debug"
            else:
                table_name = "parsing-accuracy"
            run.log({table_name: results_table})
            print("Logged to wandb...", flush=True)

        print(f"Saved results to {results_location}")
        print(f"-----------------")


def load_n_prompts(model):
    n_prompts_configs = [
        20
    ]
    # doesn't matter if we draw many
    # when taking nn as result
    if model == "nearest-neighbor" in model:
        n_prompts_configs = [1]
    return n_prompts_configs


def load_model(dset, guided_decoding, model):
    """Loads the model"""
    print("Initializing model...", flush=True)
    if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
        gin.parse_config(f"ExplainBot.parsing_model_name = '{model}'")
    elif sys.platform.startswith('win32') or sys.platform.startswith('cygwin'):
        gin.parse_config(f"ExplainBot.parsing_model_name = {model}")
    else:
        raise OSError("Unknown operating system!")

    gin.parse_config(f"ExplainBot.use_guided_decoding = {guided_decoding}")

    gin.parse_config("get_few_shot_predict_f.device = 'cuda'")

    # Case for NN and few shot gpt models
    bot = ExplainBot()

    def get_parse_text(user_input_to_parse):
        includes_all_words = None
        try:
            with torch.no_grad():
                _, result_parse_text, includes_all_words = bot.compute_parse_text(user_input_to_parse,
                                                                                error_analysis=True)
            import gc
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            result_parse_text = f"Exception: {e}, likely OOM"
        return result_parse_text, includes_all_words
    return bot, get_parse_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gd", action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--id", type=str, required=True, help="a unique id to associate with the run")
    parser.add_argument("--down_sample", action="store_true", help="this will break each run on 10 samples")
    parser.add_argument("--program_only", action="store_true", help="only uses the program name for templates")
    parser.add_argument("--subset", choices=["train", "dev", "test", "user"])
    args = parser.parse_args()

    if args.wandb:
        import wandb
        run = wandb.init(project="project-ttm", entity="dslack")
    pst = timezone('US/Pacific')
    sa_time = datetime.now(pst)
    time = sa_time.strftime('%Y-%m-%d_%H-%M')
    if args.wandb:
        wandb.run.name = f"{args.id}-{safe_name(args.model)}_{args.dataset}_gd-{args.gd}"

    main()
