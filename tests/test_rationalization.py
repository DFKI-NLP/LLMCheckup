# import os
# import sys
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
# from actions.explanation.rationalize import rationalize_operation
#
# conversation = CONVERSATION
# dataset_name = conversation.describe.get_dataset_name()
#
#
# def test_rationalization():
#     """Test rationalization functionality"""
#     parse_text = ["filter", "id", "150", "and", "rationalize", "[E]"]
#
#     return_s, status_code = rationalize_operation(conversation, parse_text, 0)
#
#     file_html = open(f"./tests/html/rationalization/{dataset_name}_rationalization.html", "w")
#     text = TEXT
#     text += return_s
#     text += "</body></html>"
#     file_html.write(text)
#
#     # Saving the data into the HTML file
#     file_html.close()
#     assert status_code == 1
