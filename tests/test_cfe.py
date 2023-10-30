# import sys
# import os
#
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
# from actions.explanation.cfe import counterfactuals_operation
#
# conversation = CONVERSATION
# dataset_name = conversation.describe.get_dataset_name()
#
#
# def test_cfe():
#     """Test cfe functionality"""
#     parse_text = ["filter", "id", "23", "and", "newcfe", "[E]"]
#
#     return_s, status_code = counterfactuals_operation(conversation, parse_text, 1)
#
#     file_html = open(f"./tests/html/cfe/{dataset_name}_cfe.html", "w")
#     text = TEXT
#     text += return_s
#     text += "</body></html>"
#     file_html.write(text)
#
#     # Saving the data into the HTML file
#     file_html.close()
#     assert status_code == 1
