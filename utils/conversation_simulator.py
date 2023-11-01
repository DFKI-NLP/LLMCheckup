"""
This file is needed for creating a conversation object for testing.
"""
from logic.conversation import Conversation
from logic.dataset_description import DatasetDescription
from logic.utils import read_and_format_data

TEXT = "<!DOCTYPE html><html><head></head><body>"

import gin


@gin.configurable()
def create_conversation(class_names, dataset_objective, dataset_description,
                        model_description, name,
                        dataset_file_path,
                        dataset_index_column,
                        target_variable_name,
                        categorical_features,
                        numerical_features,
                        remove_underscores,
                        model_file_path):
    conversation = Conversation(class_names=class_names)
    datasetDescription = DatasetDescription(
        dataset_objective=dataset_objective,
        dataset_description=dataset_description,
        model_description=model_description, name=name)

    conversation.describe = datasetDescription

    dataset, y_values, categorical, numeric = read_and_format_data(dataset_file_path,
                                                                   dataset_index_column,
                                                                   target_variable_name,
                                                                   categorical_features,
                                                                   numerical_features,
                                                                   remove_underscores)
    conversation.add_dataset(dataset, y_values, categorical, numeric)

    conversation.build_temp_dataset()


    # if conversation.describe.get_dataset_name() == 'daily_dialog':
    #     model = DANetwork()
    # else:
    #     model = load_hf_model(model_file_path, name)
    # decoder = Decoder(parsing_model_name="EleutherAI/gpt-neo-2.7B")
    # conversation.decoder = decoder

    # conversation.add_var('model', model, 'model')

    return conversation


gin.parse_config_file('./configs/test_boolq.gin')
# gin.parse_config_file('./configs/test_olid.gin')
# gin.parse_config_file('./configs/test_da.gin')

CONVERSATION = create_conversation()
