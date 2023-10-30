# coding: cp1252
import json

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_few_shot_prompt(instance):
    original_text = [
                    # "@USER You are an individual of excellent taste.",
                    #  "@USER @USER Liberals didn't give Juanita Broderick a chance to be heard or Keith Ellison's accuser. Can't have it both ways.",
                    #  "@USER @USER Yeah, poor conservatives for having eyes that see, and ears that hear. If you're loving your brain bath , then continue on , pretty soon pedophlia will be your next great virtue to champion as you see kind loving pedo's dawn your TV set to make you march in line and resist.",
                    #  "@USER @USER We need gun control!",
                    #  "@USER White Nationalist and Nazis arent the same. Also at this point I'd prefer nazis tbh",
                     "@USER Can I take you to lunch when you are in Germany?! beauty",
                     "@USER Cause who bitches be playing with :loudly_crying_face: :loudly_crying_face: :loudly_crying_face: :loudly_crying_face: god aint give me the gift for nothing",
                     "@USER @USER He should just stick to sports but Liberals can't help themselves.",
                     "@USER Good riddance.",
                     "@USER I've seen people with some the most beautiful (not to mention original) art i last my eyes on. but only have 70 followers because they get buried by half-assed bullshit. all because they're not pals with someone popular",
                     "@USER Zarries didn’t start this conversation! It was a larry shipper that trash talked our ship!! So how about y’all petty ass larries stay out of our mention eh?!!",
                     "@USER @USER Wow!!! Never thought I'd see the left actually admitting they want to shut down free speech unless it's their own. At least they are admitting they are the facist. After all they did create the 2 most facist groups...ANTIFA and KKK.",
                     "@USER I can’t think of a worse 3 game start for a coach at such a big time university in my life. 1-2 and almost lose to Fred Samford University lol I know it’s Sanford. But yea they still owe him whatever he is owed",
                     "@USER I’m so sorry for what you are dealing with. Hope the next place will be much better! :smiling_face_with_smiling_eyes:",
                     "@USER @USER He is complicit in the abuse of young men. Why would he care about a woman. He make me sick.",
                     "@USER @USER He is losing his mind",
                     "@USER OMG! Did you see that “almost” victim had a chance to shoot the “incompetent” criminal with a gun. We need better gun control laws.",
                     "@USER - both don’t support Medicare for all - both voted against the dreamers - both voted for the racist police brutality bill - both voted to give Donald trump and even larger military budget . - both didn’t do a thing on gun control - both voted for the racist payday loans bill",
                     "@USER @USER From some resist dork that probably refers to the POTUS as frump or drump or tfrump? I will refer to her however I please. Frankly she is Ms. Psychotic Leftist Activist Lunatic Dr. Ford. Better now?",
                     "@USER He is too bitchy... And you?! I ask a question and you dont respond Why would this be 1 way street?"
                     ]
    augmented_text = [
                    # "You have an impeccable sense of style.",
                    #   "@USER @USER When it comes to Juanita Broderick and Keith Ellison's accuser, liberals didn't provide an equal opportunity to be heard. It's a contradiction.",
                    #   "@USER @USER It's disheartening to witness conservatives being criticized for their perceptive nature and ability to listen. If you're content with your current perspective, brace yourself for the possibility that pedophilia might be the next controversial virtue to be championed, with sympathetic and affectionate pedophiles appearing on your television, encouraging conformity and resistance.",
                    #   "@USER @USER Advocating for gun control is crucial!",
                    #   "@USER There is a distinction between White Nationalist and Nazis. However, I would rather have neither at this point.",
                      "@USER Can we arrange to have lunch together during your visit to Germany? It would be delightful!",
                      "@USER Because some people are just toying with :loudly_crying_face: :loudly_crying_face: :loudly_crying_face: :loudly_crying_face:; God didn't grant me this gift for nothing.",
                      "@USER @USER It would be better if he focused solely on sports, but Liberals always find it hard to resist other topics.",
                      "@USER Glad to see them go.",
                      "@USER I've come across artists with the most stunning (and truly unique) art that has mesmerized me. Unfortunately, they only have 70 followers as their work gets overshadowed by subpar content. It's disheartening to see how popularity dictates recognition rather than true talent.",
                      "@USER Zarries didn't initiate this conversation! It was a larry shipper who criticized our ship!! So, could you all, petty larries, please refrain from interfering in our mentions, eh?!!",
                      "@USER @USER Wow!!! I never expected the left to openly acknowledge their desire to suppress free speech, except for their own. At least they are admitting to their stance. However, it's worth noting that they associate with the two most controversial groups: ANTIFA and KKK.",
                      "@USER I can't recall a more disappointing start for a coach at a prestigious university in my life. With a record of 1-2 and struggling against Fred Samford University (lol, I know it's Sanford), it's clear there's much owed to him regardless.",
                      "@USER I empathize with what you're going through. Wishing you a much better experience at your next place! :smiling_face_with_smiling_eyes:",
                      "@USER @USER He is implicated in the abuse of young men. It's unlikely he would care about a woman. He disgusts me.",
                      "@USER @USER He's going crazy.",
                      "@USER OMG! Did you witness the 'almost' victim having an opportunity to shoot the 'incompetent' criminal with a gun? It highlights the need for improved gun control laws.",
                      "@USER - Neither of them backs Medicare for all - both voted against the dreamers - both supported the racist police brutality bill - both voted to increase Donald Trump's military budget - neither took action on gun control - both voted for the racist payday loans bill.",
                      "@USER @USER Coming from someone who likely calls the POTUS 'frump' or 'drump' or 'tfrump,' I will address her as I see fit. Frankly, she is Ms. Psychotic Leftist Activist Lunatic Dr. Ford. Satisfied?",
                      "@USER He is too moody... And you?! I ask a question and you don't respond. Why is this a one-way street?"
                      ]

    prompt = f"Each 2 items in the following list contains the original text and rephrased text.\n"
    for i in range(len(original_text)):
        prompt += f"original text: {original_text[i]}\nrephrased text: {augmented_text[i]})\n"
    prompt += f"original text: {instance}\nrephrased text: "

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
            res = decoded_generation.split("\n")[-1].split("rephrased text:")[1]
        except:
            num_error += 1
            print(num_error)
    print({
        "idx": idx,
        "rephrased text": res
    })

    json_list.append({
        "idx": idx,
        "rephrased text": res
    })

jsonString = json.dumps(json_list)
jsonFile = open(f"/netscratch/qwang/data_augmentation_gpt-neo-2.7b.json", "w")
jsonFile.write(jsonString)
jsonFile.close()

