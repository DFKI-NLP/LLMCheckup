# LLMCheckup
Dialogical Interpretability Tool for LLMs

## Models:
In our study, we identified three LLMs for our purposes.

- LLama2 (https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- Mistral (https://huggingface.co/mistralai/Mistral-7B-v0.1)
- Stable Beluga 2 (finetuned llama2-70B) (https://huggingface.co/petals-team/StableBeluga2)

### Deployment:
We support different methods for deployment:
- Original models
- Quantized by [GPTQ](https://arxiv.org/abs/2210.17323)
- Quantized by [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and load in 8 bits
- peer2peer by [petals](https://github.com/bigscience-workshop/petals)

## Use case :
### Fact checking
Dataset: COVID-Fact

Link: https://github.com/asaakyan/covidfact

### Commonsense Question Answering
Dataset: ECQA

Link: https://github.com/dair-iitd/ECQA-Dataset
