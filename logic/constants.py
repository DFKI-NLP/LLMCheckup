operations_with_id = ["show", "predict", "likelihood", "similar", "nlpattribute", "rationalize", "cfe", "augment"]

deictic_words = ["this", "that", "it", "here"]

confirm = ["Yes", "Yes, please", "Ok", "I would like to see it", "Please show it to me", "Fine", "Yes, thank you.", "Sounds good", "Yes, I do.", "I would like to see it, thanks."]
disconfirm = ["No, thank you", "Nope", "Sorry, no", "No, I want to ask something different", "That's it, thanks!", "No, that's enough."]


thanks = ["Thanks!", "OK!", "I see", "Thanks a lot!", "Thank you.", "Alright, thank you!", "That's nice, thanks a lot :)", "Good, thanks!", "Thank you very much.", "Looks good, thank you!", "Great, thank you very much!", "Ok, thanks!", "Thank you for the answer."]
bye = ["Goodbye!", "Bye-bye!", "Bye!", "Ok, bye then!", "That's all, bye!", "See you next time!", "Thanks for the chat, bye!"]

dialogue_flow_map = {
    "thanks": ["You are welcome!", "No problem.", "I'm glad I could help.", "Can I help you with something else?","Is there anything else I could do for you?"],
    "bye": ["Goodbye!", "Bye-bye!", "Have a nice day!", "See you next time!"],
    "sorry": ["Sorry! I couldn't understand that. Could you please try to rephrase?", "My apologies, I did not get what you mean.", "I'm sorry but could you rephrase the message, please?", "I'm not sure I can do this. Maybe you have another request for me?", "This is likely out of my expertise, can I help you with something else?", "This was a bit unclear. Could you rephrase it, please?", "Let's try another option. I'm afraid I don't have an answer for this."]
    }

user_prompts = ["Ok, just let me know if you want to know anything else.", "Alright.", "I see, just tell me if you want to know about some other operation."]

# compatible operations that we need to generate suggestions
metadata = ["show", "data", "data train", "data test", "countdata", "label", "model"]
prediction = ["predict", "score", "mistake count", "mistake sample"]
understanding = ["similar", "keywords"]
explanation = ["nlpattribute topk", "rationalize"]
perturbation = ["cfe", "augment"]
#TODO add choices for nlpattribute
nlpattribute_methods = ["integrated_gradients", " attention", " lime", "input_x_gradient"]
valid_operation_names = metadata + prediction + understanding + explanation + perturbation
valid_operation_names_words = []
# adding individual words for the cases like "nlpattribute sentence"
for w in valid_operation_names:
    valid_operation_names_words.extend(w.split())
operation2set = dict()
for op in metadata:
    operation2set[op] = metadata
for op in prediction:
    operation2set[op] = prediction
for op in understanding:
    operation2set[op] = understanding
for op in explanation:
    operation2set[op] = explanation
for op in perturbation:
    operation2set[op] = perturbation

no_filter_operations = ["data", "data train", "data test", "countdata", "label", "model", "mistake count", "mistake sample", "keywords"]

map2suggestion = {
    "show": "Would you like me to show a sample?",
    "data": "Do you want to learn more about the data?",
    "data train": "I can provide a description of the training data if you like.",
    "data test": "I can provide a description of the test data if you like.",
    "countdata": "Would you like to know about the dataset statistics?",
    "label": "I can also tell you more about the labels.",
    "model": "Should I explain the underlying model in more detail?",
    "predict": "Do you want to see a prediction?",
    "score": "I can tell you more about the scores.",
    "mistake count": "Should I also show the mistake count?",
    "mistake sample": "Do you want to have a look at some sample mistakes?",
    "similar": "Would you like to see a similar instance?",
    "keywords": "I can also show you the top keywords.",
    "nlpattribute topk": "Shall I show you the importance scores?",#TODO
    "rationalize": "Would you like to see a rationale?",
    "cfe": "Should I generate a counterfactual?",
    "augment": "Should I augment this instance?"
    }
