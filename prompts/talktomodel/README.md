These prompts from TTM are not yet fully rewritten and also need an implementation in *actions*.


## likelihood

*likelihood* operation combined with various filters

### includes_+_likelihood.txt
### lengthfilter_+_likelihood.txt


## replace

See Issue #74

### change.txt
### change_comparisons.txt
### previousfilter_+_change.txt
### previousoperation_+_change.txt


## spanimportance
An *explanation* operation that, based on the pre-computed feature attributions, would retrieve *k* instances where a custom input span has the highest attribution score. The sorting could be influenced by (1) the rank of the span in sorted attributions and (2) the absolute difference to the next highest attributed word/sentence.

`spanimportance topk {k}`

### important_cat_features.txt
### important_num_features.txt
