operations_with_id = ["show", "predict", "likelihood", "similar", "nlpattribute", "rationalize", "cfe", "augment"]

deictic_words = ["this", "that", "it", "here"]

confirm = ["Yes", "Of course", "I agree", "Correct", "Yeah", "Right", "That's what I meant", "Indeed", "Exactly", "True"]
disconfirm = ["No", "Nope", "Sorry, no", "I think there is some misunderstanding", "Not right", "Incorrect", "Wrong", "Disagree"]

thanks = ["Thanks!", "OK!", "I see", "Thanks a lot!", "Thank you.", "Alright, thank you!", "That's nice, thanks a lot :)", "Good, thanks!", "Thank you very much.", "Looks good, thank you!", "Great, thank you very much!", "Ok, thanks!", "Thank you for the answer."]
bye = ["Goodbye!", "Bye-bye!", "Bye!", "Ok, bye then!", "That's all, bye!", "See you next time!", "Thanks for the chat, bye!"]

dialogue_flow_map = {
    "thanks": ["You are welcome!", "No problem.", "I'm glad I could help.", "Can I help you with something else?","Is there anything else I could do for you?"],
    "bye": ["Goodbye!", "Bye-bye!", "Have a nice day!", "See you next time!"],
    "sorry": ["Sorry! I couldn't understand that. Could you please try to rephrase?", "My apologies, I did not get what you mean.", "I'm sorry but could you rephrase the message, please?", "I'm not sure I can do this. Maybe you have another request for me?", "This is likely out of my expertise, can I help you with something else?", "This was a bit unclear. Could you rephrase it, please?", "Let's try another option. I'm afraid I don't have an answer for this."]
    }

