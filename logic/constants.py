operations_with_id = ["show", "predict", "likelihood", "similar", "nlpattribute topk", "nlpattribute integrated_gradients", "nlpattribute attention", "nlpattribute lime", "nlpattribute input_x_gradient", "rationalize", "cfe", "augment"]

deictic_words = ["this", "that", "it", "here"]

confirm = ["Yes", "Yes, please", "Yes, thanks", "Ok", "Ok, thank you", "Ok, thanks", "I would like to see it", "Please show it to me", "Fine", "Yes, thank you.", "Sounds good", "Yes, I do", "I would like to see it, thanks.", "Yes, I'd like to see it."]
disconfirm = ["No", "No, thank you", "No, thanks", "Nope", "Sorry, no", "No, I want to ask something different", "No, that's it", "No, I would like to see another operation.", "That's it, thanks!", "No, that's enough.", "I want to ask something else", "I don't need it, thank you", "I am interested in another operation", "Not now, thanks"]


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
explanation = ["nlpattribute topk", "nlpattribute integrated_gradients", "nlpattribute attention", "nlpattribute lime", "nlpattribute input_x_gradient", "rationalize"]
perturbation = ["cfe", "augment"]
qatutorial = ["qacfe", "qada", "qafa", "qarationale", "qasim", "qacustominput"]
valid_operation_names = metadata + prediction + understanding + explanation + perturbation + qatutorial
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
for op in qatutorial:
    operation2set[op] = qatutorial

no_filter_operations = ["data", "data train", "data test", "countdata", "label", "model", "mistake count", "mistake sample", "keywords", "qacfe", "qada", "qafa", "qarationale", "qasim", "qacustominput"]

map2suggestion = {
    'show': ['Should I provide you with a sample?', 'Would you like to see this sample from the dataset?', 'Should I show the sample?', 'Would you like to see the instance from the dataset?', 'I can also show you the sample from the data.', 'I can show this example, if you like.'],
    'data': ['Are you interested in learning more about the data?', 'Are you curious about the data?', 'Would you like to know more about the data?', 'Are you interested in finding out more about the data?', 'Would you like to gain a deeper understanding of the data?', 'Would you like to explore the data?'],
    'data train': ["If you're interested, I can give you a rundown of the training data.", 'I can give you an overview of the training data.', "If you're interested, I can provide a breakdown of the training data.", 'I can elucidate the training data in detail.', "If you're interested, I can go over the training data and present it to you.", 'The training data can be explained by me if you want.'],
    'data test': ["If you're interested, I can give you an overview of the test data.", 'The test data can be explained by me if you want.', "If you're interested, I can elucidate the test data.", 'I can explain how the test data looks like.', "If you're interested, I can introduce the test data.", "If you're interested, I can provide a description of the test data."],
    'countdata': ['Are you interested in obtaining statistics from this dataset?', 'Would you like to have an overview of the statistics of this dataset?', 'Would you like to have more information about the statistics of the dataset?', 'Would you like to know how many data points are available?', 'Are you interested in accessing the statistical details about the data?', 'Are you interested in accessing the statistics of the dataset?'],
    'label': ['I can provide a more in-depth explanation of the labels.', 'I can give you a rundown of the labels as well.', 'You can also find out more about the labels.', 'I can provide a more in-depth explanation of the labels.', 'I can provide a more detailed account of the labels.', 'Additionally, I have more information on the labels.'],
    'model': ['Is it necessary to elaborate on the underlying model?', 'Do I need to elaborate on the underlying model?', 'Is it necessary for me to elaborate on the underlying model?', 'Is it necessary to clarify the underlying model?', 'Do I need to elaborate on the model?', 'Is it necessary to clarify the underlying model?'],
    'predict': ['Would you like to see a prediction?', 'Would you like to get a prediction?', 'Are you interested in seeing a prediction?', 'I can also show a prediction.', 'If you are interested, I can show the prediction.'],
    'score': ['I also have some knowledge about the scores.', 'Would you like to know more about the scores?', 'The scores can be explained by me in greater detail.', 'I have a summary of the scores.', 'I have a summary of the scores for you.', 'The scores can be explained by me as well.'],
    'mistake count': ['Is it necessary to disclose the number of errors also?', 'Would it be beneficial to disclose the number of mistakes the model has made?', 'Is it necessary to disclose the number of errors too?', 'Would it be beneficial to show you the number of errors also?', 'Is it necessary to report the number of errors also?', 'Would you like to know how many mistakes the model makes?'],
    'mistake sample': ['Should I provide you with some instances of errors?', 'Would you you like to see some mistakes that the model makes?', 'I can offer some examples of mistakes as well.', 'Should I give you a few illustrations of errors?', 'I can show you some incorrectly predicted examples.', 'Would you like to see some examples of the mistakes?'],
    'similar': ['Do you want to see some similar examples?', 'Are you interested in seeing other examples like this?', 'Would you be interested in seeing other similar instances?', 'I can show you similar samples from the data.', 'Should I show you some similar examples?', 'Do you want to see a similar example?'],
    'keywords': ['I am capable of presenting you with the most sought-after keywords.', 'I can present you the most effective keywords.', 'The keywords that are most important can also be displayed by me.', 'The keywords that are most crucial can also be shown by me.', 'The top keywords can also be displayed by me.', 'I can demonstrate the key words as well.', 'I can demonstrate the key words also.'],
    'nlpattribute topk': ['Would you like to view the importance scores?', 'Would you like to see the importance scores?', 'Would you like to access the importance scores?', 'Do you wish to view the importance scores?', 'Are you curious about the importance scores?', 'Are you looking for the feature attributions?'],
    'nlpattribute integrated_gradients': ['Would you like to view the importance scores computed with the integrated gradient method?', 'Should I show you the integrated gradient of the importance scores?', 'Are you interested in accessing the integrated gradient of importance scores?', 'Would you like to view the integrated gradients of importance scores?', 'Are you interested in examining integrated gradients to determine importance scores?', 'Are you interested in observing integrated gradients and obtaining importance scores?'],
    'nlpattribute attention': ['Would you like to see the importance scores of attention?', 'Would you like to view the attention scores of importance?', 'Would you like to view the importance scores based on attention?', 'Would you like to know the importance scores of attention?', 'Would you like to access the importance scores of attention?', 'Would you like to access the attention scores for importance?', 'Are you interested in accessing the importance scores with attention?', 'Would you like to access the attention-based importance scores?'],
    'nlpattribute lime': ['Are you interested in accessing the importance scores with LIME?', 'Would you like to access the importance scores based on LIME?', 'Would you like to view the importance scores based on the LIME method?', 'Are you interested in viewing the importance scores computed with LIME?', 'Would you like to view the importance scores derived from LIME?', 'Would you like to view the importance scores based on LIME?'],
    'nlpattribute input_x_gradient': ['Would you like to view the importance scores on an input with the input x gradient?', 'Would you like to view the importance scores based on the input x gradient?', 'Are you interested in accessing the importance scores with an input x gradient?', 'Would you like to view the attributions based on the input x gradient?', 'Would you like to view the importance scores as they are computed with an input x gradient?'],
    'rationalize': ['Would you like to see a rationalization?', 'Would you like to see a rationale for this?', 'I can also generate a rationalization.', 'Should I provide an explanation in natural language?', 'Would you like to see an explanation based on the rationale generation?', 'Should I generate an explanation?'],
    'cfe': ['Would you like to see a counterfactual?', 'I can show you a counterfactual as well.', 'Would you be interested in seeing a counterfactual for this?', 'Should I generate a counterfactual?', 'I can also show you how to flip the prediction for this instance.', 'Would you like to have a look at a counterfactual?'],
    'augment': ['Should I augment this sample?', 'I can also do the data augmentation.', 'Would you like to have this instance augmented?', 'Should I perform the data augmentation?', 'Would you like to see the augmented data?', 'Should I augment this sample?'],
    'qacfe': ['Do you have an interest in more details on the counterfactual operations?', 'Do you want to find out how the counterfactual works?', 'Are you interested in learning about the counterfactuals?', 'Are you interested in understanding how counterfactual operation works?', 'Are you interested in learning about the counterfactuals?', 'Should I also explain the counterfactual operation?'],
    'qada': ['Would it be beneficial to introduce the data augmentation operation?', 'Is it necessary to introduce the data augmentation operation as well?', 'Is it necessary to introduce the data augmentation operation?', 'Should I introduce the data augmentation operation as well?', 'I can introduce the data augmentation operation too, if you wish.', 'Would you like to know how the data augmentation operation works?'],
    'qafa': ['Would you like to know more about the methods of feature attributions?', 'Would you like to see the feature attribution methods?', 'Would you like to know about the token-level importance scores and feature attributions?', 'Would you like to learn more about the feature attribution methods?', 'Shall I also explain different feature attribution methods?'],
    'qarationale': ['I can also explain how the rationalization works.', 'I can also explain how the rationales are generated.', 'Would you like to know more about the rationalization process?', 'Should I provide explanations for the rationalization operation as well?'],
    'qasim': ['Do you want to learn about the similarity operation?', 'Would you like to learn about the similarity operation?', 'Should I explain the similarity operation as well?', 'I can also provide explanations for the similarity operation.', 'Are you interested in learning more about the similarity operation?'],
    'qacustominput': ['Would you like to know what the custom input means?', "Should I also explain the custom input setting?", "Should I provide explanations for the custom input as well?", "Would you like to learn more about the custom input setting?", "I can also introduce the custom input setting."]
    }
