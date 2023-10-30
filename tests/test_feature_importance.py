# import sys
# import os
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
# from actions.explanation.feature_importance import feature_importance_operation
#
# """
# pytest -q test_feature_importance.py           (in tests folder)
#
# pytest -q test_feature_importance.py --global  (under root folder)
# """
#
# conversation = CONVERSATION
# dataset_name = conversation.describe.get_dataset_name()
#
#
# def test_feature_importance():
#     """
#     Test feature importance for a single instance with given id
#     """
#
#     parse_text = ["filter", "id", "33", "and", "nlpattribute", "topk", "1", "[E]"]
#
#     return_s, status_code = feature_importance_operation(conversation, parse_text, 1, False)
#
#     file_html = open(f"./tests/html/feature_importance/{dataset_name}_feature_importance.html", "w")
#     text = TEXT
#     text += return_s
#     text += "</body></html>"
#     file_html.write(text)
#
#     # Saving the data into the HTML file
#     file_html.close()
#
#     assert status_code == 1
#
#
# def test_multiple_feature_importance():
#     """
#     Test feature importance for multiple instances with given ids
#     """
#
#     parse_text = ["filter", "id", "33", "or", "id", "151", "and", "nlpattribute", "topk", "2", "[E]"]
#
#     return_s, status_code = feature_importance_operation(conversation, parse_text, 1, False)
#
#     file_html = open(f"./tests/html/feature_importance/{dataset_name}_multiple_feature_importance.html", "w")
#     text = TEXT
#     text += return_s
#     text += "</body></html>"
#     file_html.write(text)
#
#     # Saving the data into the HTML file
#     file_html.close()
#
#     assert status_code == 1
#
#
# def test_feature_importance_with_custom_input():
#     """
#     Test feature importance for custom input
#     """
#
#     parse_text = ["nlpattribute", "topk", "3"]
#
#     conversation.custom_input = "conservatives left frustrated as Congress passes big spending bills"
#     conversation.used = False
#     return_s, status_code = feature_importance_operation(conversation, parse_text, 1, False)
#
#     file_html = open(f"./tests/html/feature_importance/{dataset_name}_feature_importance_with_custom_input.html", "w")
#     text = TEXT
#     text += return_s
#     text += "</body></html>"
#     file_html.write(text)
#
#     # Saving the data into the HTML file
#     file_html.close()
#
#     assert status_code == 1
#
#
# def test_feature_importance_all():
#     """
#     Test feature importance
#     """
#
#     parse_text = ["filter", "id", "53", "and", "nlpattribute", "all", "[E]"]
#
#     return_s, status_code = feature_importance_operation(conversation, parse_text, 1, False)
#
#     file_html = open(f"./tests/html/feature_importance/{dataset_name}_feature_importance_all.html", "w")
#     text = TEXT
#     text += return_s
#     text += "</body></html>"
#     file_html.write(text)
#
#     # Saving the data into the HTML file
#     file_html.close()
#
#     assert status_code == 1
#
#
# def test_feature_importance_sentence_level():
#     """
#     Test feature importance sentence level
#     """
#
#     parse_text = ["filter", "id", "15", 'or', 'filter', 'id', '20', "nlpattribute", "sentence", "[E]"]
#
#     return_s, status_code = feature_importance_operation(conversation, parse_text, 0, False)
#
#     file_html = open(f"./tests/html/feature_importance/{dataset_name}_feature_importance_sentence_level.html", "w")
#     text = TEXT
#     text += return_s
#
#     text += "</body></html>"
#     file_html.write(text)
#
#     # Saving the data into the HTML file
#     file_html.close()
#
#     assert status_code == 1
#
#
# def test_feature_importance_with_custom_input_at_sentence_level():
#     """
#     Test feature importance for custom input at sentence level
#     """
#
#     parse_text = ["nlpattribute", "sentence"]
#
#     conversation.custom_input = "conservatives left frustrated as Congress passes big spending bills. he left the house and go to the bank."
#     conversation.used = False
#     return_s, status_code = feature_importance_operation(conversation, parse_text, 1, False)
#
#     file_html = open(f"./tests/html/feature_importance/{dataset_name}_feature_importance_with_custom_inputat_sentence_level.html", "w")
#     text = TEXT
#     text += return_s
#     text += "</body></html>"
#     file_html.write(text)
#
#     # Saving the data into the HTML file
#     file_html.close()
#
#     assert status_code == 1
