"""Contains a function returning a dict mapping the key word for actions to the function.

This function is used to generate a dictionary of all the actions and the corresponding function.
This functionality is used later on to determine the set of allowable operations and what functions
to run when parsing the grammar.
"""
from actions.about.function import function_operation
from actions.about.self import self_operation
from actions.context.followup import followup_operation
from actions.context.last_turn_filter import last_turn_filter
from actions.context.last_turn_operation import last_turn_operation
from actions.explanation.feature_importance import feature_importance_operation
from actions.explanation.rationalize import rationalize_operation
from actions.filter.filter import filter_operation
from actions.filter.includes_token import includes_operation
from actions.metadata.count_data_points import count_data_points
from actions.metadata.data_summary import data_operation
from actions.metadata.define import define_operation
from actions.metadata.labels import show_labels_operation
from actions.metadata.model import model_operation
from actions.metadata.show_data import show_operation
from actions.nlu.keyword import keyword_operation
from actions.nlu.similarity import similar_instances_operation
from actions.perturbation.augment import augment_operation
from actions.perturbation.cfe import counterfactuals_operation
from actions.prediction.mistakes import show_mistakes_operation
from actions.prediction.predict import predict_operation
from actions.prediction.random_prediction import random_prediction
from actions.prediction.score import score_operation
from actions.qatutorial.tutorial import tutorial_operation


def get_all_action_functions_map():
    """Gets a dictionary mapping all the names of the actions in the parse tree to their functions."""
    actions = {
        'countdata': count_data_points,
        'filter': filter_operation,
        'predict': predict_operation,
        'randompredict': random_prediction,
        'self': self_operation,
        'previousfilter': last_turn_filter,
        'previousoperation': last_turn_operation,
        'data': data_operation,
        'keywords': keyword_operation,
        'followup': followup_operation,
        'show': show_operation,
        'model': model_operation,
        'function': function_operation,
        'score': score_operation,
        'label': show_labels_operation,
        'mistake': show_mistakes_operation,
        'define': define_operation,
        'predictionfilter': filter_operation,
        'labelfilter': filter_operation,
        'includes': includes_operation,
        'similar': similar_instances_operation,
        'rationalize': rationalize_operation,
        'nlpattribute': feature_importance_operation,
        'cfe': counterfactuals_operation,
        'augment': augment_operation,
        'qatutorial': tutorial_operation
    }
    return actions