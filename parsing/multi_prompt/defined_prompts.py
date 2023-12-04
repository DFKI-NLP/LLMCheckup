operation_type_prompt = "Choose one of the following operations: function | self | nlpattribute | rationalize | countdata | data | model | label | show | keywords | similar | augment | cfe | mistake | predict | randompredict | score | qacfe | qada | qafa | qarationale | qasim\nExamples:\nInput: What can you do? Output: function\nInput: Can you tell me what is this application about? Output: self\nInput: What are the most important words in the sample 427? Output: nlpattribute\nInput: Show me the token attributions for id 74. Output: filter id 74 and nlpattribute all.\nInput: What are the most important tokens according to the attention scores for id 122? Output: filter id 122 and nlpattribute attention.\nInput: Can you explain this prediction for id 61 with input gradients? Output: filter id 61 and nlpattribute input_x_gradient\nInput: Explain sample 25 with the integrated gardients method. Output: filter id 25 and nlpattribute integrated_gradient\nInput: Please provide a rationale for id 25. Output: rationalize\nInput: How many data points are in the dataset? Output: countdata\nInput: What are the data? Output: data\nInput: Can you explain me the model? Output: model\nInput: Which labels do we have? Output: label\nInput: Show id 22 Output: show\nInput: What are the top keyword tokens? Output: keywords\nInput: What are the most common keywords in the data? Output: keywords\nInput: Show me another instance that is similar to id 148. Output: similar\nInput: Augment id 46. Output: augment\nInput: Generate a counterfactual for id 45. Output: cfe\nInput: What is the counterfactual for this sample? Output: cfe\nInput: Show me a cfe for the sample with id 100. Output: cfe\nInput: How can I flip the prediction for id 1290? Output: cfe\nInput: Show me some mistakes. Output: mistake\nInput: What is the model's prediction on id 82? Output: predict\nInput: Please pre-compute 7 random samples. Output: randompredict 7\nInput: Please tell me about the accuracy scores. Output: score\nInput: What's counterfactual operation in NLP? Output: qacfe \nInput: How does counterfactual operation work (in general)? Output: qacfe \nInput: Explain me the data augmentation process Output: qada \nInput: Can you provide an explanation for the data augmentation method in the context of NLP? Output: qada \nInput: What role does feature attribution play in NLP? Output: qafa \nInput: How does feature attribution work? Output: qafa \nInput: What's rationalization in NLP? Output: qarationale \nInput: Explain the rationalization operation Output: qarationale \nInput: What's semantic similarity? Output: qasim \nInput: Definition of semantic similarity? Output: qasim\nPlease choose only one operation from the list separated by bars: function | self | nlpattribute | rationalize | countdata | data | model | label | show | keywords | similar | augment | cfe | mistake | predict | randompredict | score | qacfe | qada | qafa | qarationale | qasim\n"#Select one operation from this list for the user input

valid_operation_names = ["function", "self", "nlpattribute", "rationalize", "countdata", "data", "model", "label", "show", "keywords", "similar", "augment", "cfe", "mistake", "predict", "randompredict", "score", "qacfe", "qada", "qafa", "qarationale", "qasim"]

valid_operation_meanings = ["function", "describe itself", "feature importance, token attribution", "rationalization", "count data points", "dataset", "model", "labels", "show sample", "keywords", "similar", "augment", "counterfactual", "count or show mistakes", "predict instance", "pre-compute samples", "performance scores", "what is counterfactual operation, explain", "what is data augmentation operation, explain", "what is feature attribution operation, explain", "what is rationalization operation, explain", "what is similarity operation, explain"]

valid_operation_prompt_samples = ["What is the function of this application?", "Tell me about LLMCheckUp", "What are the most attributed tokens for id 122?", "Can you provide a rationale for id 489?", "How many data points are there in total?", "Explain to me the dataset, please", "What is the underlying model?", "Which labels do we have?", "Please show sample number 15.", "Top 10 keywords in the data.", "What are other similar examples for id 68?", "Can you perform data augmentation for id 9555?", "Please find a counterfactual for id 110.", "What are the mistakes that the model makes?", "Can you show me the prediction that the model makes on id 77?", "What are the scores?", "What is the counterfactual operation meant for?", "How does data augmentation work?", "How do you compute the feature attributions?", "How do you generate a rationale?", "What is the similarity operation, how does it work?"]

operation2attributes = {"nlpattribute": ["all", "topk", "input_x_gradient", "integrated_gradients", "lime", "attention"],
                        "keywords": ["all"],
                        "mistake": ["count", "sample"],
                        "score": ["accuracy", "f1", "micro", "macro", "weighted", "precision", "recall"],
                        "randompredict": []# numbers
                       }

operations_wo_attributes = ["function", "self", "countdata", "data", "model", "label"]

tutorial_operations = ["qacfe", "qada", "qafa", "qarationale", "qasim"]

tutorial2operation = {"qacfe":"cfe", "qada":"augment", "qafa":"nlpattribute", "qarationale":"rationalize", "qasim":"similar"}

operation2tutorial = {v: k for k,v in tutorial2operation.items()}

# add "\nInput: input text Output:" at the end of each prompt 
nlpattribute_prompt = "Please parse the input as shown in the examples:\nInput: most important features for data point 1000 Output: filter id 1000 and nlpattribute all [E]\nInput: explain id 15 using lime Output: filter id 15 and nlpattribute lime [E]\nInput: why do you predict instance with id 987 this way, can you explain it using the input gradients? Output: filter id 987 and nlpattribute input_x_gradient [E]\nInput: 10 most important features for id's 5 regarding attention Output: filter id 5 and nlpattribute topk 10 attention [E]\nInput: Why does the model predict id 178 like this? How does the attention-based explanation look like? Show me the top 5 features Output: filter id 178 and nlpattribute topk 5 attention [E]\nInput: what three features most influence the model's predictions for ids 1515 using lime Output: filter id 1515 and nlpattribute topk 3 lime [E]\nShow me the key seven features for id 799 using attention for token attribution. Output: filter id 799 and nlpattribute topk 7 attention [E]"

rationalize_prompt = "Please parse the input as shown in the examples:\nInput: explain id 150 in natural language Output: filter id 150 and rationalize [E]\nInput: explain id 6390 with rationale Output: filter id 6390 and rationalize [E]\nInput: generate a natural language explanation for id 2222 Output: filter id 2222 and rationalize [E]\nInput: rationalize the prediction for id 9555 Output: filter id 9555 and rationalize [E]"

show_prompt = "Please parse the input as shown in the examples:\nInput: Can you display the instance with id 2451? Output: filter id 2451 and show [E]\nInput: For 3315, please show me the values of the features. Output: filter id 3315 and show [E]\nInput: Could you show me data point number 215? Output: filter id 215 and show [E]\nInput: Show id 105111, please. Output: filter id 105111 and show [E]"

keywords_prompt = "Please parse the input as shown in the examples:\nInput: What are the most frequent keywords in the data? Output: keywords all [E]\nInput: Keywords Output: keywords all most [E]\nInput: What are the three most frequent keywords? Output: keywords 3 [E]\nInput: Which five words occur the most in the data? Output: keywords 5 [E]\nInput: Which word occur least in the data? Output: keywords 1 least [E]"

similar_prompt = "Please parse the input as shown in the examples:\nInput: Please retrieve an example that is similar to ID 50 Output: filter id 50 and similar 1 [E]\nInput: Could you give me an analogous data point to ID 75. Output: filter id 75 and similar 1 [E]\nInput: I'm looking for a case that is akin to id 14. Could you help me with that? Output: filter id 14 and similar 1 [E]\nInput: Show 3 similar instances to ID 25. Output: filter id 25 and similar 3 [E]\nInput: Can you bring up 3 instances that shares similarities with ID 35? Output: filter id 35 and similar 3 [E]\nInput: Could you locate 6 comparable data point to ID 75 for me? Output: filter id 75 and similar 6 [E]"

augment_prompt = "Please parse the input as shown in the examples:\nInput: Please augment id 25 Output: filter id 25 and augment [E]\nInput: Please create a new instance based on id 50 Output: filter id 50 and augment [E]\nInput: Starting from id 75, how would a new instance look like? Output: filter id 75 and augment [E]"

cfe_prompt = "Please parse the input as shown in the examples:\nInput: What does instance with id 22 need to do to change the prediction? Output: filter id 22 and cfe [E]\nInput: show me cfe's for the instance with id 22 Output: filter id 22 and cfe [E]\nInput: How would you flip the prediction for id 23? Output: filter id 23 and cfe [E]\nInput: How do I change the prediction for the data point with id number 34? Output: filter id 34 and cfe [E]\nInput: What is the way to change the prediction for the data point with the id number 422? Output: filter id 422 and cfe [E]\nInput: Could you please tell me the predictions for id 5132 and what you have to do to flip the prediction? Output: filter id 54 and predict and cfe [E]"

mistake_prompt = "Please parse the input as shown in the examples:\nInput: can you show me how much data the model predicts incorrectly? Output: mistake count [E]\nInput: tell me the amount of data the model predicts falsely Output: mistake count [E]\nInput: How frequently does the model make mistakes? Output: mistake count [E]\nInput: show me the number of data points the model forecasts inaccurately? Output: mistake count [E]\nInput: show me data the model gets wrong Output: mistake sample [E]\nInput: what are some data points you get incorrect? Output: mistake sample [E]\nInput: could you show me a few examples of data that you get wrong? Output: mistake sample [E]"

predict_prompt = "Please parse the input as shown in the examples:\nInput: What do you predict for 215? Output: filter id 215 and predict [E]\nInput: What is the prediction for data point number 9130? Output: filter id 9130 and predict [E]\nInput: For id 776, please provide the prediction. Output: filter id 776 and predict [E]\nInput: predict 320 Output: filter id 320 and predict [E]\nInput: return prediction id 13423 Output: filter id 13423 and predict [E]\nInput: please display the prediction of the instance with id 34 Output: filter id 34 and predict [E]"

randompredict = "Please parse the input as shown in the examples:\nInput: Please do the pre-computation for 5 samples randomly. Output: randompredict 5 [E]\nInput: Pre-compute 5 randomly selected instances from the dataset. Output: randompredict 5 [E]"

#default value: score default [E]
score_prompt = "Please parse the input as shown in the examples:\nInput: testing accuracy Output: score accuracy [E]\nInput: give me the accuracy on the data Output: score accuracy [E]\nInput: could you give me the test accuracy on the training data? Output: score accuracy [E]\nInput: how often are you correct? Output: score accuracy [E]\nInput: what's the rate you do correct predictions? Output: score accuracy [E]\nInput: how accurate is the model on all the data? Output: score accuracy [E]\nInput: nice! could you give me the test f1? Output: score accuracy f1 [E]\nInput: display score Output: score default [E]\nInput: testing f1 Output: score f1 [E]\nInput: I meant what is the f1 score on the evaluation data Output: score f1 [E]\nInput: What is the micro-F1 score? Output: score f1 micro [E]\nInput: What is the macro-F1 score? Output: score f1 macro [E]\nInput: What is the weighted F1 score? Output: score f1 weighted [E]\nInput: can you show me the precision on the data? Output: score precision [E]\nInput: What is the micro precision? Output: score precision micro [E]\nInput: What is the macro score for precision? Output: score precision macro [E]\nInput: Please compute the weighted precision. Output: score precision weighted [E]\nInput: give the recall score Output: score recall [E]\nInput: What is the micro recall? Output: score recall micro [E]\nInput: What is the macro recall? Output: score recall macro [E]\nInput: What is the weighted recall? Output: score recall weighted [E]\nInput: can you show me the roc score on the testing data? Output: score roc [E]\nInput: what's the roc score Output: score roc [E]"

operation2prompt = {"nlpattribute": nlpattribute_prompt,
                    "rationalize": rationalize_prompt,
                    "show": show_prompt,
                    "keywords": keywords_prompt,
                    "similar": similar_prompt,
                    "augment": augment_prompt,
                    "cfe": cfe_prompt,
                    "mistake": mistake_prompt,
                    "predict": predict_prompt,
                    "randompredict": randompredict,
                    "score": score_prompt
                   }

operation_needs_id = ["nlpattribute", "rationalize", "show", "similar", "augment", "cfe", "predict"]