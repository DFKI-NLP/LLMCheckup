import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
from actions.metadata.show_data import show_operation

conversation = CONVERSATION
dataset_name = conversation.describe.get_dataset_name()


def test_show_data():
    """Test show data functionality"""
    parse_text = ["show", "[E]"]

    return_s, status_code = show_operation(conversation, parse_text, 1)

    file_html = open(f"./tests/html/show/{dataset_name}_show_data.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1
