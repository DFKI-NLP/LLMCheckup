# pytest framework 
If you want to run a single test file, then:
```bash
pytest tests/test_xxx.py 
```

Otherwise, it will run all test files containing in the **tests** folder:
```bash 
pytest tests 
```

## The test template
You could follow the template below to write your own test cases(you have to create a test file under **_/tests_**):
```py
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION

# from your_operation_file import test_operation

conversation = CONVERSATION


def test_your_function():
    # List your parsed test
    parse_text = ["[E]"]

    # Call your defined operation
    return_s, status_code = test_operation(conversation, parse_text, 1)

    # Create a folder called your_folder and give a name to html file
    file_html = open(f"./tests/html/your_folder/your_test_function.html", "w")
    text = TEXT
    text += return_s
    text += "</html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1

```