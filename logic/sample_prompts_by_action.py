"""Sample a prompt for a specific action."""
import numpy as np

from logic.prompts import get_user_part_of_prompt

ACTION_2_FILENAME = {
    # About
    "self": "describe_self.txt",
    "function": "describe_function.txt",
    # Metadata
    "show": "show.txt",
    "model": "describe_model.txt",
    "data_summary": "describe_data.txt",
    "countdata": "count_data.txt",
    "labels": "label.txt",
    # Prediction
    "predict": "predict.txt",
    "random_predict": ["random_predict.txt", "random_predict_chatgpt.txt"],
    "score": "score.txt",
    "mistake_count": "mistake_count.txt",
    "mistake_sample": "mistake_sample.txt",
    # Understanding
    "similar": ["similar.txt", "similar_chatgpt.txt"],
    "keyword": "keywords.txt",
    # Explanation
    "explain": ["local_feature_importance.txt", "local_feature_importance_chatgpt.txt"],
    "rationalization": ["rationalize.txt", "rationalize_chatgpt.txt"],
    # Custom input
    "custom_input_prediction": ["custom_input_prediction_chatgpt.txt", "custom_input_prediction.txt"],
    "custom_input_feature_importance": ["custom_input_feature_importance_chatgpt.txt", "custom_input_feature_importance.txt"],
    "c_rationale": ["custom_input_rationalize.txt", "custom_input_rationalize_chatgpt.txt"],
    "c_cfe": ["custom_input_cfe.txt", "custom_input_cfe_chatgpt.txt"],
    "c_data": ["custom_input_data_augmentation.txt", "custom_input_data_augmentation_chatgpt.txt"],
    "c_sim": "custom_input_similarity.txt",
    # Perturbation
    "cfe": "cfe.txt",
    "augment": ["augmentation_chatgpt.txt", "augmentation.txt"],
    # QA
    "qa_cfe": "qacfe.txt",
    "qa_da": "qada.txt",
    "qa_fa": "qafeature_attribution.txt",
    "qa_rat": "qarationale.txt",
    "qa_sim": "qasim.txt",
    "qa_ci": "qacustominput.txt"
}


def replace_non_existent_id_with_real_id(prompt: str, real_ids: list[int]) -> str:
    """Attempts to replace ids that don't exist in the data with ones that do.

    The problem is that prompt generation may create prompts with ids that don't occur
    in the data. This is fine for the purposes of generating training data. However,
    when the user clicks "generate an example question" we should try and make sure
    that the example question uses ids that actually occur in the data, so they
    don't get errors if they try and run the prompt (this seems a bit unsatisfying).
    This function resolves this issues by trying to find occurrences of ids in the prompts
    and replaces them with ids that are known to occur in the data.

    Arguments:
        prompt: The prompt to try and replace with an actual id in the data.
        real_ids: A list of ids that occur in the data **in actuality**
    Returns:
        resolved_prompt: The prompt with ids replaced
    """
    split_prompt = prompt.split()
    for i in range(len(split_prompt)):
        if split_prompt[i] in ["point", "id", "number", "for", "instance"]:
            if i+1 < len(split_prompt):
                # Second option is for sentence end punctuation case
                if split_prompt[i+1].isnumeric() or split_prompt[i+1][:-1].isnumeric():
                    split_prompt[i+1] = str(np.random.choice(real_ids))
    output = " ".join(split_prompt)
    return output


def sample_prompt_for_action(action: str,
                             filename_to_prompt_ids: dict,
                             prompt_set: dict,
                             conversation) -> str:
    """Samples a prompt for a specific action.

    Arguments:
        prompt_set: The full prompt set. This is a dictionary that maps from **prompt** ids
                    to info about the prompt.
        action: The action to sample a prompt for
        filename_to_prompt_ids: a map from the prompt filenames to the id of a prompt.
                                Note, in this context, 'id' refers to the 'id' assigned
                                to each prompt and *not* an id of a data point.
    Returns:
        prompt: The sampled prompt
    """

    real_ids = conversation.get_training_data_ids()
    if action == "self":
        return "Could you tell me a bit more about what this is?"
    elif action == "function":
        return "What can you do?"
    elif action in ACTION_2_FILENAME:
        filename_end = ACTION_2_FILENAME[action]

        if type(filename_end) != str:
            chosen_id = np.random.choice(len(filename_end))
            filename_end = filename_end[chosen_id]

        for filename in filename_to_prompt_ids:
            if ("includes" not in filename) and filename.endswith(filename_end):

                if conversation.used:
                    if "custom_input" in filename:
                        continue

                prompt_ids = filename_to_prompt_ids[filename]
                chosen_id = np.random.choice(prompt_ids)
                i = 0
                # Try to not return prompts that are not complete
                # for the particular dataset (i.e., those that have
                # a wildcard still in them with "{" )
                prompt = prompt_set[chosen_id]["prompts"][0]
                user_part = get_user_part_of_prompt(prompt)
                while "{" in user_part or i < 100:
                    chosen_id = np.random.choice(prompt_ids)
                    i += 1
                    prompt = prompt_set[chosen_id]["prompts"][0]
                    user_part = get_user_part_of_prompt(prompt)
                final_user_part = replace_non_existent_id_with_real_id(user_part, real_ids)
                return final_user_part

            elif ("includes" in filename) and ("includes" in filename_end) and filename.endswith(filename_end):
                prompt_ids = filename_to_prompt_ids[filename]
                chosen_id = np.random.choice(prompt_ids)
                prompt = prompt_set[chosen_id]["prompts"][0]
                user_part = get_user_part_of_prompt(prompt)
                return user_part
        message = f"Unable to filename ending in {filename_end}!"
        raise NameError(message)
    else:
        message = f"Unknown action {action}"
        raise NameError(message)
