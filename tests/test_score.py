import sys
import os



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
from actions.prediction.score import score_operation

conversation = CONVERSATION
dataset_name = conversation.describe.get_dataset_name()


def test_score_accuracy():
    """Test score accuracy functionality"""
    parse_text = ["score", "accuracy", "[E]"]

    return_s, status_code = score_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/score/{dataset_name}_score_accuracy.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1


def test_score_default():
    """Test score default functionality"""
    parse_text = ["score", "default", "[E]"]

    return_s, status_code = score_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/score/{dataset_name}_score_default.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1


def test_score_f1():
    """Test score f1 functionality"""
    if dataset_name != 'daily_dialog':
        parse_text = ["score", "f1", "[E]"]
    else:
        parse_text = ["score", "f1", "weighted", "[E]"]

    return_s, status_code = score_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/score/{dataset_name}_score_f1.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1


def test_score_precision():
    """Test score precision functionality"""
    if dataset_name != 'daily_dialog':
        parse_text = ["score", "precision", "[E]"]
    else:
        parse_text = ["score", "precision", "weighted", "[E]"]
    return_s, status_code = score_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/score/{dataset_name}_score_precision.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1


def test_score_recall():
    """Test score recall functionality"""
    if dataset_name != 'daily_dialog':
        parse_text = ["score", "recall", "[E]"]
    else:
        parse_text = ["score", "recall", "weighted", "[E]"]
    return_s, status_code = score_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/score/{dataset_name}_score_recall.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1


def test_score_roc():
    """Test score roc functionality"""
    parse_text = ["score", "roc", "[E]"]

    return_s, status_code = score_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/score/{dataset_name}_score_roc.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1
