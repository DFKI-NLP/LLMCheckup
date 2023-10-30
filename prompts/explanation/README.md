# Explanation operations

## Global Feature Importance

Action defined in explanation/topk.py

### global_feature_importance.txt
* 21x `important all [E]`
* 11x `important topk [E]`

### global_feature_importance_class.txt
* 9x `important {class_names} [E]`


## Local Feature Importance

Action defined in explanation/feature_importance.py

### local_feature_importance.txt
* 10x `filter id and nlpattribute all [E]`
* 4x `filter id and nlpattribute sentence [E]`
* 5x `filter id and predict and nlpattribute all [E]`
* 6x `filter id and nlpattribute topk [E]`

### local_feature_importance_chatgpt.txt
**GPT-4 generated**
* 6x `filter id and nlpattribute all [E]`
* 9x `filter id and nlpattribute topk [E]`
* 3x `filter id or filter id and nlpattribute topk [E]`
* 2x `filter id or filter id or filter id and nlpattribute topk [E]`

### custom_input_feature_importance.txt
* 1x `nlpattribute [E]`
* 1x `nlpattribute sentence [E]`
* 1x `nlpattribute topk [E]`

### custom_input_feature_importance_chatgpt.txt
**GPT-4 generated**
* 7x `nlpattribute [E]`
* 7x `nlpattribute sentence [E]`
* 6x `nlpattribute topk [E]`


## Rationalization

Action defined in explanation/rationalize.py

### rationalize.txt
* 4x `filter id and rationalize [E]`

### rationalize_chatgpt.txt
**GPT-4 generated**
* 20x `filter id and rationalize [E]`
