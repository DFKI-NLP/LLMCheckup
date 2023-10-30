import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
from actions.prediction.prediction_likelihood import predict_likelihood

conversation = CONVERSATION
dataset_name = conversation.describe.get_dataset_name()


def test_prediction_likelihood():
    """Test prediction likelihood functionality"""
    parse_text = ["filter", "id", "2215", "and", "likelihood", "[E]"]

    return_s, status_code = predict_likelihood(conversation, parse_text, 1)

    file_html = open(f"./tests/html/likelihood/{dataset_name}_prediction_likelihood.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1
