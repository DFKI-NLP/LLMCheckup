import traceback

from flask import Flask

app = Flask(__name__)


def get_few_shot_prompt(instance):
    labels = [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
    original_text = [
        "@USER I'm not surprised. It takes a sick woman to make this type of accusations. The kind you can only find on the far left. Liberals are praying that the really weak members of the GOP like Collins and Flake will pull back their support.",
        "@USER @USER @USER Rivera is shit. He is the last one left. Our problems run through him. The day we fire @USER the better off we will be",
        "@USER Guilty?",
        "@USER Thank you for your fortitude and perseverance. Please give McConnell a kick in the butt from some of us conservatives.",
        "@USER Well you are and we love you",
        "@USER If he survived he obviously doesnt know anything about gun control. Are we sure he is a lib?",
        "@USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER But that interferes with antifa’s goals! Because they want to be feared?",
        "@USER @USER @USER @USER @USER I know you are not being facetious. You are incapable of understanding contradictions.",
        "@USER This is just more of the standard Trudeau “Liberals” emotional manipulation of Canadians. They think Canadians are stupid.",
        "@USER Fuck it. Just do it Haha",
        "@USER @USER Lmao saying you are happy for someone publically doesn't mean you magically stop having feelings for them and are over them. Have you ever had a bf before? :skull: And for the last time nobody is saying she should have stayed with him. It's the messiness immediately after",
        "@USER @USER @USER :face_with_tears_of_joy: :face_with_tears_of_joy: :face_with_tears_of_joy: He got our asses Not even believing the truth about the truth!! Lmao! &amp; We Know! Water is damn wet...but cant even believe it coming out his lying ass mouth! :see-no-evil_monkey: :see-no-evil_monkey:",
        "*waits for a buncha braindead rednecks to burn the Levi's that they're wearing and get hauled off to the ER en masse* URL",
        "@USER Or gun control. Or both.",
        "@USER He was saying niggas who say they boycotting Nike for the military are racist because the military fuck with Nike :face_with_tears_of_joy:",
        "@USER Well she supported $7 co payment for Dr visits when President of AMA thats an indication . She is Ambitious thats all .",
        "Scottish @USER @USER running out of excuses and lies to justify her position URL",
        "@USER An ugly old man trying to make himself relevant again with more lies Remember him saying he had a deathbed confession from CIA director casey? Never happened according to the bodyguards in that room shouldn't that be enough to question this morons credibility",
        "@USER i like him he is cute",
        "@USER No time for political correctness. It is a Lie. Everyone knows this. Behind the scenes the liberals are laughing. Take the vote.",
        "@USER @USER @USER Good for you.. he is bit miss guided for me to follow as the guy to me seems more like an idiot &amp; a crock but if you think that he is perfect even though there is multiple investigations against him whom am I to disagree. I'm still looking for a better role model for America cya",
        "@USER @USER No one gives a shit and no one will miss her! Sorry @USER will! URL",
        "@USER An idiot. Where the fuck do they get these people?",
        "@USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER Antifa tell the truth?",
        "@USER Do really even care what comes out the his foolish and annoying mouth? Go home David and let the adults MAGA"
    ]
    counterfactuals = [
        "@USER I'm not surprised. It takes a compassionate woman to raise these concerns. The kind you can find on the far right. Liberals are hoping that the strong members of the GOP will stand up for their values.",
        "@USER @USER @USER Rivera is excellent. He is the last one left. Our solutions flow through him. The day we promote @USER, the better off we will be.",
        "@USER Innocent?",
        "@USER Thank you for your lack of determination and perseverance. Please give McConnell a pat on the back from some of us liberals.",
        "@USER Well you aren't and we don't love you.",
        "@USER If he survived, he obviously knows a lot about gun control. Are we sure he isn't a conservative?",
        "@USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER But that aligns perfectly with antifa's goals! Because they want to promote understanding?",
        "@USER @USER @USER @USER @USER I know you are being facetious. You are exceptionally skilled at understanding contradictions.",
        "@USER This is just more of the standard Trudeau 'Conservatives' rational persuasion of Canadians. They believe Canadians are smart.",
        "@USER Think it through. Just don't do it Haha",
        "@USER @USER Lmao saying you are unhappy for someone publicly doesn't mean you magically start having feelings for them and are still into them. Have you ever had a bf before? :skull: And for the last time nobody is saying she should have left him. It's the clarity immediately after.",
        "@USER @USER @USER :face_with_tears_of_joy: :face_with_tears_of_joy: :face_with_tears_of_joy: He got our backs, even believing the truth about the truth!! Lmao! & We Know! Water is damn dry...but can't even believe it coming out his honest mouth! :see-no-evil_monkey: :see-no-evil_monkey:",
        "waits for a group of enlightened intellectuals to cherish the Levi's they're wearing and enjoy a day of camaraderie URL",
        "@USER Or gun rights. Or neither.",
        "@USER He was saying people who say they boycotting Nike for the military are not racist because the military supports Nike :face_with_tears_of_joy:",
        "@USER Well, she opposed $7 co-payment for Dr visits when President of AMA, that's an indication. She is Compassionate, that's all.",
        "Scottish @USER @USER finding new reasons and truths to support her stance URL",
        "@USER A ugly old man trying to maintain his relevance with more truths. Remember him saying he had a deathbed confession from CIA director casey? Verified by the bodyguards in that room, shouldn't that be enough to establish this person's credibility?",
        "@USER i dislike him he is unattractive",
        "@USER Plenty of time for political correctness. It is the Truth. Everyone knows this. Behind the scenes, the conservatives are laughing. Postpone the vote.",
        "@USER @USER @USER Good for you.. he is a bit misguided for me to follow as the guy to me seems more like a genius & honest but if you think that he is an idiot even though there is multiple investigations against him whom am I to disagree. I'm still looking for a worse role model for America cya",
        "@USER @USER Everyone cares deeply and everyone will miss her! Sorry, @USER won't! URL",
        "@USER A genius. Where the hell do they find such exceptional individuals?",
        "@USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER Antifa lie?",
        "@USER Do you even care what comes out of his wise and uplifting mouth? Stay, David, and let the adults MAGA."
    ]

    # prompt = f"Each 2 items in the following list contains the original text and the counterfactual that presents the opposite perspective.\n"
    # for i in range(len(original_text)):
    #     prompt += f"original text: {original_text[i]}\n counterfactual: {counterfactuals[i]})\n"
    # prompt += f"original text: {instance}\n counterfactual:"
    prompt = f"Each 2 items in the following list contains the original text and the counterfactual. Counterfactual changes the prediction.\n"
    for i in range(len(original_text)):
        prompt += f"original text: {original_text[i]}\n counterfactual: {counterfactuals[i]})\n"
    prompt += f"original text: {instance}\n counterfactual:"

    return prompt


def counterfactuals_operation(conversation, parse_text, i, **kwargs):
    if conversation.custom_input is not None and conversation.used is False:
        instance = conversation.custom_input
    else:
        assert len(conversation.temp_dataset.contents["X"]) == 1

        try:
            idx = conversation.temp_dataset.contents["X"].index[0]
        except ValueError:
            return "Sorry, invalid id", 1

        instance = conversation.get_var("dataset").contents["X"].iloc[idx]["text"]

    model = conversation.decoder.gpt_model
    tokenizer = conversation.decoder.gpt_tokenizer

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Get predictions
    pred_model = conversation.get_var('model').contents
    model_prediction = pred_model.predict(None, idx, conversation)[0]

    few_shot = True
    # few_shot = False

    temp = instance.split(" ")
    res = []
    for i in temp:
        if i != "@USER":
            res.append(i)
    instance = " ".join(res)

    if few_shot:
        if conversation.custom_input is not None and conversation.used is False:
            prompt = get_few_shot_prompt(conversation.custom_input)
            app.logger.info("Custom input is used")
        else:
            prompt = get_few_shot_prompt(instance)
            app.logger.info("Processing few-shot prompt")
    else:
        # app.logger.info("Processing zero-shot think step-by-step prompt")
        # prompt = f"Given instance: '{instance}'. It is predicted as {conversation.class_names[model_prediction]}. " \
        #          f"Based on the instance, please generate a counterfactual that flips the prediction to " \
        #          f"{conversation.class_names[1-model_prediction]}. Let's think step by step."
        prompt = f"Given instance: '{instance}'. It is predicted as {conversation.class_names[model_prediction]}. Based on the instance, please " \
                 f"generate a new example but with prediction {conversation.class_names[1-model_prediction]}."

    inputs = tokenizer(prompt, return_tensors="pt")

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    length = len(instance.split(" "))

    # if torch.cuda.is_available():
    #     device = "cuda"
    #     model.to(device)
    #     input_ids = input_ids.to(device=device)
    #     attention_mask = attention_mask.to(device)
    # else:
    #     device = "cpu"
    if few_shot:
        generation = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_new_tokens=length * 2,
            no_repeat_ngram_size=2,
            temperature=0.95,
            top_p=0.95
        )
    else:
        generation = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=2048,
            do_sample=True,
            no_repeat_ngram_size=2,
            temperature=0.8,
            top_p=0.8
        )

    decoded_generation = tokenizer.decode(generation[0], skip_special_tokens=True)

    try:
        if few_shot:
            decoded_generation = decoded_generation.split("\n")[-1]
            res = decoded_generation.split('counterfactual:')[-1]
        else:
            res = decoded_generation.split(prompt)[1]
    except Exception as e:
        print(f"Generation failed due to {traceback.format_exc()}")
        return "Generation failed", 1

    return_s = ""
    return_s += f"<b>Original text:</b> (ID {idx})<br>" + instance + "<br>"
    return_s += f"The prediction is <span style=\"background-color: #6CB4EE\">{conversation.class_names[model_prediction]}</span>."
    return_s += "<br><br>"

    if few_shot:
        return_s += "<b>Counterfactual:</b><br>" + res + "<br>"
    else:
        return_s += f"Counterfactual result (reasoning step by step): {res}"

    return return_s, 1
