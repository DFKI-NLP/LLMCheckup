These prompts need an implementation in *actions*.

## globaltopk combinations
Right now, `globaltopk` only operates on a fixed set of pre-computed global explanations. It is only possible to use `important` on the whole dataset or on a specific class (`{classname}`).
If there's a way to efficiently compute them on other temp_datasets, we can combine this operation with different filters:

### includes_+_globaltopk.txt
* 24x `includes and important all [E]`
* 3x `includes and important topk [E]`


## lengthfilter
See Issue #52

Most prompts are adapted from TTM's `{num_features}` and `{num_values}`.

### lengthfilter_+_countdata.txt
* 2x `lengthfilter chars and countdata [E]`
* 2x `lengthfilter sents and countdata [E]`
* 3x `lengthfilter words and countdata [E]`

### lengthfilter_+_globaltopk.txt
* 2x `lengthfilter chars and important all [E]`
* 2x `lengthfilter sents and important all [E]`
* 3x `lengthfilter words and important all [E]`
* 1x `lengthfilter chars and lengthfilter chars and important all [E]`
* 1x `lengthfilter words and predict and important all [E]`
* 2x `lengthfilter words and show and important all [E]`
* 2x `includes and lengthfilter sents and important all [E]`
* 3x `includes and lengthfilter words and important all [E]`
* 2x `predictionfilter and lengthfilter chars and important all [E]`
* 2x `predictionfilter and lengthfilter sents and important all [E]`
* 4x `predictionfilter and lengthfilter words and important all [E]`
* 1x `previousfilter and lengthfilter words and important all [E]`

### lengthfilter_+_predict.txt
* 3x `lengthfilter chars and predict [E]`
* 5x `lengthfilter sents and predict [E]`
* 4x `lengthfilter words and predict [E]`
* 1x `lengthfilter chars and show and predict [E]`
* 2x `lengthfilter sents and show and predict [E]`
* 1x `lengthfilter words and show and predict [E]`
* 2x `includes and lengthfilter chars and predict [E]`
* 2x `includes and lengthfilter sents and predict [E]`
* 5x `includes and lengthfilter words and predict [E]`
* 2x `includes or lengthfilter words and predict [E]`
* 3x `previousfilter and lengthfilter words and predict [E]`

### lengthfilter_+_score.txt
* 1x `lengthfilter chars and score accuracy [E]`
* 1x `lengthfilter sents and score accuracy [E]`
* 1x `lengthfilter words and score accuracy [E]`

### lengthfilter_+_show.txt
* 1x `lengthfilter chars and show [E]`
* 1x `lengthfilter sents and show [E]`
* 2x `lengthfilter words and show [E]`

## likelihood

### likelihood_class.txt
* 2x `likelihood [E]`
