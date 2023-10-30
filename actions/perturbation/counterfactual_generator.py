# coding: cp1252
import json

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_few_shot_prompt(instance):
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
        "@USER No time for political correctness. It is a Lie. Everyone knows this. Behind the scenes the liberals are laughing. Take the vote."

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
        "@USER Plenty of time for political correctness. It is the Truth. Everyone knows this. Behind the scenes, the conservatives are laughing. Postpone the vote."
    ]

    prompt = f"Each 2 items in the following list contains the original text and the counterfactual. Counterfactual changes the prediction.\n"
    for i in range(len(original_text)):
        prompt += f"original text: {original_text[i]}\n counterfactual: {counterfactuals[i]})\n"
    prompt += f"original text: {instance}\n counterfactual:"

    return prompt

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

df = pd.read_csv("/home/qwang/InterroLang/data/offensive_val.csv")

num_error = 0
json_list = []

for idx in range(len(list(df["text"]))):
    print(idx)
    res = "error"
    instance = df["text"][idx]
    # golden_label = df["label"][idx]

    temp = instance.split(" ")
    instance_list = []
    for i in temp:
        if i != "@USER":
            instance_list.append(i)
    instance = " ".join(instance_list)

    prompt = get_few_shot_prompt(instance)

    inputs = tokenizer(prompt, return_tensors="pt")

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    input_ids = input_ids.to(device=device)
    attention_mask = attention_mask.to(device)

    length = len(instance.split(" "))

    generation = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        max_new_tokens=length * 2,
        no_repeat_ngram_size=2,
        temperature=0.95,
        top_p=0.95
    )

    decoded_generation = tokenizer.decode(generation[0], skip_special_tokens=True)

    try:
        res = decoded_generation.split(prompt)[1]
    except:
        try:
            res = decoded_generation.split("\n")[-1].split("counterfactual:")[1]
        except:
            num_error += 1
            print(num_error)
    print({
        "idx": idx,
        "counterfactual": res
    })

    json_list.append({
        "idx": idx,
        "counterfactual": res
    })

jsonString = json.dumps(json_list)
jsonFile = open(f"/netscratch/qwang/counterfactual_gpt-neo-2.7b.json", "w")
jsonFile.write(jsonString)
jsonFile.close()

