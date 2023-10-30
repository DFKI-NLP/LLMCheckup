import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
from actions.metadata.count_data_points import count_data_points

conversation = CONVERSATION
dataset_name = conversation.describe.get_dataset_name()


def test_countdata():
    """Test countdata functionality"""
    parse_text = ["countdata", "[E]"]

    return_s, status_code = count_data_points(conversation, parse_text, 0)

    file_html = open(f"./tests/html/ndatapoints/{dataset_name}_ndatapoints.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1
