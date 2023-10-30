import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
from actions.metadata.labels import show_labels_operation

conversation = CONVERSATION
dataset_name = conversation.describe.get_dataset_name()


def test_labels():
    """Test label functionality"""
    parse_text = ["label", "[E]"]

    return_s, status_code = show_labels_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/labels/{dataset_name}_labels.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1
