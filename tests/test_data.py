import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
from actions.metadata.data_summary import data_operation

conversation = CONVERSATION
dataset_name = conversation.describe.get_dataset_name()


def test_data_summary():
    """Test data summary functionality"""
    parse_text = ["data", "[e]"]

    return_s, status_code = data_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/data/{dataset_name}_data_summary.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1
