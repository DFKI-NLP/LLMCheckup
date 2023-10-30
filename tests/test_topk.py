import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
from actions.explanation.global_topk import global_top_k

conversation = CONVERSATION
dataset_name = conversation.describe.get_dataset_name()


def test_topk():
    """Test topk functionality"""
    parse_text = ["important", "all", "[E]"]

    return_s, status_code = global_top_k(conversation, parse_text, 0)

    file_html = open(f"./tests/html/topk/{dataset_name}_topk.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1


def test_topk_with_value():
    """Test topk functionality"""
    parse_text = ["important", "10", "[E]"]

    # return_s, status_code = global_top_k(conversation, parse_text, 0)
    return_s, status_code = global_top_k(conversation, parse_text, 0)
    file_html = open(f"./tests/html/topk/{dataset_name}_topk_with_value.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1
