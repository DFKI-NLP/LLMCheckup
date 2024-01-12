"""The app main."""
import uuid
from datetime import datetime
from os.path import isfile, join

import gin
import json
import logging
import os
import traceback
import random

from flask import Flask
from flask import render_template, request, Blueprint
from logging.config import dictConfig

from actions.prediction.predict import convert_str_to_options
from actions.util_functions import text2int, get_current_prompt
from logic.core import ExplainBot
from logic.sample_prompts_by_action import sample_prompt_for_action

import easyocr
import numpy as np
from scipy.io.wavfile import read
import librosa
import soundfile as sf

import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

my_uuid = uuid.uuid4()


# gunicorn doesn't have command line flags, using a gin file to pass command line args
@gin.configurable
class GlobalArgs:
    def __init__(self, config, baseurl):
        self.config = config
        self.baseurl = baseurl


# Parse gin global config
gin.parse_config_file("global_config.gin")

# Get args
args = GlobalArgs()

bp = Blueprint('host', __name__, template_folder='templates')

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Try to use all fragmented GPU memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

# Parse application level configs
gin.parse_config_file(args.config)

# Setup the explainbot
BOT = ExplainBot()


@bp.route('/')
def home():
    """Load the explanation interface."""
    app.logger.info("Loaded Login")
    objective = BOT.conversation.describe.get_dataset_objective()

    BOT.conversation.build_temp_dataset()

    df = BOT.conversation.temp_dataset.contents['X']
    f_names = list(BOT.conversation.temp_dataset.contents['X'].columns)

    dataset = BOT.conversation.describe.get_dataset_name()

    entries = []
    for j in range(10):
        temp = {}
        for f in f_names:
            if dataset == "ECQA" and f == "choices":
                temp[f] = convert_str_to_options(df[f][j])
            else:
                temp[f] = df[f][j]
        entries.append(temp)

    return render_template("index.html", currentUserId="user", datasetObjective=objective, entries=entries,
                           dataset=dataset)


@bp.route("/log_feedback", methods=['POST'])
def log_feedback():
    """Logs feedback"""
    feedback = request.data.decode("utf-8")
    app.logger.info(feedback)
    split_feedback = feedback.split(" || ")

    message = f"Feedback formatted improperly. Got: {split_feedback}"
    assert split_feedback[0].startswith("MessageID: "), message
    assert split_feedback[1].startswith("Feedback: "), message
    assert split_feedback[2].startswith("Username: "), message
    assert split_feedback[3].startswith("Answer: "), message

    message_id = split_feedback[0][len("MessageID: "):]
    feedback_text = split_feedback[1][len("Feedback: "):]
    username = split_feedback[2][len("Username: "):]
    answer = split_feedback[3][len("Answer: "):]

    current_time = datetime.now()
    time_stamp = current_time.timestamp()
    date_time = datetime.fromtimestamp(time_stamp)
    str_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")

    logging_info = {
        "id": message_id,
        "feedback_text": feedback_text,
        "username": username,
        "answer": answer,
        "dataset": BOT.conversation.describe.get_dataset_name(),
        "parsed_text": BOT.parsed_text,
        "user_text": BOT.user_text,
        "timestamp": str_time
    }

    BOT.log(logging_info)
    return ""


@bp.route("/export_history", methods=["Post"])
def export_history():
    BOT.export_history()
    return ""


@bp.route("/sample_prompt", methods=["Post"])
def sample_prompt():
    """Samples a prompt"""
    data = json.loads(request.data)
    action = data["action"]
    username = data["thisUserName"]

    prompt = sample_prompt_for_action(action,
                                      BOT.prompts.filename_to_prompt_id,
                                      BOT.prompts.final_prompt_set,
                                      BOT.conversation)
    logging_info = {
        "username": username,
        "requested_action_generation": action,
        "generated_prompt": prompt
    }
    BOT.log(logging_info)

    return prompt


@bp.route("/get_response", methods=['POST'])
def get_bot_response():
    """Load the box response."""
    if request.method == "POST":
        response = ""
        try:
            flag = None
            audio = None

            try:
                # Receive the uploaded image
                img = request.files["image"]
                flag = "img"
            except:
                pass
            try:
                data = json.loads(request.data)
                flag = "text"
            except:
                pass

            try:

                audio = request.files["audio"]
                audio.save("recording.wav")

                flag = "audio"
            except:
                pass

            if flag == "img":
                # Save image locally
                img.save(f"./{img.filename}")
                app.logger.info(f"Image uploaded!")

                if torch.cuda.is_available():
                    gpu = True
                else:
                    gpu = False
                reader = easyocr.Reader(['en'], gpu=gpu)  # this needs to run only once to load the model into memory

                result = reader.readtext(f"{img.filename}", detail=0)

                if len(result) < 2:
                    raise ValueError("Only one sentence is recognized. Please try other images!")
                else:
                    temp = {'first_input': result[0], 'second_input': result[1]}

                    BOT.conversation.custom_input = temp
                    BOT.conversation.used = False
                    app.logger.info(f"[CUSTOM INPUT] {temp}")
                    response = "You have given a custom input via uploaded image. " \
                               "Please enter a follow-up question or prompt! <br><br>" + "Entered custom input: <br>"
                    if BOT.conversation.describe.get_dataset_name() == "covid_fact":
                        response += f"Claim: {temp['first_input']} <br>Evidence: {temp['second_input']} <>"
                    else:
                        response += f"Question: {temp['first_input']} <br>Choices: {temp['second_input']} <>"
            elif flag == "audio":

                model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
                processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

                # Load wav in array
                x, _ = librosa.load('./recording.wav', sr=16000)
                sf.write('tmp.wav', x, 16000)

                a = read("tmp.wav")
                temp = np.array(a[1], dtype=np.float64)
                inputs = processor(temp, sampling_rate=16000, return_tensors="pt")
                generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

                # Convert words to digits if available
                user_text = text2int(transcription[0])

                BOT.user_text = user_text
                conversation = BOT.conversation

                # Remove generated wav files
                os.remove("./recording.wav")
                os.remove("./tmp.wav")

                app.logger.info("generating the bot response")
                response = f"<b>Recorded text</b>: {user_text}<br><br>"

            elif flag == "text":
                # Change level for QA
                level = data["qalevel"]
                prompt_type = data["prompt_type"]

                BOT.conversation.qa_level = level
                BOT.conversation.prompt_type = prompt_type
                app.logger.info(f"Prompt type: {prompt_type}")

                # Normal user input
                if data['custom_input'] == '0':
                    # Normal user input
                    user_text = data["userInput"]
                    BOT.user_text = user_text
                    conversation = BOT.conversation

                    app.logger.info("generating the bot response")
                    response = BOT.update_state(user_text, conversation)
                elif data['custom_input'] == '1':
                    # custom input
                    user_text = data["userInput"]

                    if BOT.conversation.describe.get_dataset_name() == "ECQA":
                        if len(user_text["second_input"].split("-")) != 5:
                            return "5 choices should be provided and concatenated by '-'!"

                    BOT.conversation.custom_input = user_text
                    BOT.conversation.used = False
                    app.logger.info(f"[CUSTOM INPUT] {user_text}")
                    response = "You have given a custom input. " \
                               "Please enter a follow-up question or prompt! <br><br>" + "Entered custom input: <br>"
                    if BOT.conversation.describe.get_dataset_name() == "covid_fact":
                        response += f"Claim: {user_text['first_input']} <br>Evidence: {user_text['second_input']} <>"
                    else:
                        response += f"Question: {user_text['first_input']} <br>Choices: {user_text['second_input']} <>"

                    BOT.conversation.store_last_parse(f"custominput '{user_text}'")
                else:
                    # custom input removal
                    app.logger.info(f"[CUSTOM INPUT] Custom input is removed!")
                    BOT.conversation.custom_input = None
                    BOT.conversation.used = True
                    response = "Entered custom input is now removed! <>"
                BOT.write_to_history(BOT.user_text, response)
        except torch.cuda.OutOfMemoryError:
            response = "I recognized a CUDA out of memory. I suggest to choose a smaller " \
                       "model for your hardware configuration or with GPTQ/4bit quantization. You can do that by " \
                       "opening the global_config.gin file and editing the value of GlobalArgs.config to an " \
                       "equivalent with a model of smaller parameter " \
                       "size, e.g. \"ecqa_llama_gptq.gin\" or \"ecqa_pythia.gin\"."
        except Exception as ext:
            app.logger.info(f"Traceback getting bot response: {traceback.format_exc()}")
            app.logger.info(f"Exception getting bot response: {ext}")
            response = random.choice(BOT.dialogue_flow_map["sorry"])
        return response


@bp.route("/custom_input", methods=["Post"])
def custom_input():
    data = json.loads(request.data)
    custom_input = data["custom_input"]
    username = data["thisUserName"]

    BOT.conversation.custom_input = custom_input
    BOT.conversation.used = False

    app.logger.info("custom_input: " + custom_input)

    return custom_input


@bp.route("/filter_dataset", methods=["POST"])
def filter_dataset():
    filter_text = json.loads(request.data)["filterMsgText"]
    df = BOT.conversation.stored_vars["dataset"].contents["X"]
    if len(filter_text) > 0:
        filtered_df = df[df[BOT.text_fields].apply(lambda row: row.str.contains(filter_text)).any(axis=1)]

        BOT.conversation.temp_dataset.contents["X"] = filtered_df
        app.logger.info(f"{len(filtered_df)} instances of {BOT.conversation.describe.dataset_name} include the filter "
                        f"string '{filter_text}'")
        final_df = filtered_df
    else:
        final_df = df
    return {
        'jsonData': final_df.to_json(orient="index"),
        'totalDataLen': len(df)
    }


@bp.route("/reset_temp_dataset", methods=["Post"])
def reset_temp_dataset():
    data = json.loads(request.data)
    username = data["thisUserName"]

    # Reset the tempdataset
    BOT.conversation.build_temp_dataset()

    app.logger.info("Reset temp dataset successfully!")

    return "reset temp_dataset"


@bp.route("/get_prompt", methods=["Post"])
def get_prompt():
    return get_current_prompt(BOT.parsed_text, BOT.conversation)


app = Flask(__name__)
app.register_blueprint(bp, url_prefix=args.baseurl)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == "__main__":
    # clean up storage file on restart
    app.logger.info(f"Launching app from config: {args.config}")
    app.run(debug=False, port=4455, host="localhost")
