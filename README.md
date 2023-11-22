# LLMCheckup
![Static Badge](https://img.shields.io/badge/python-3.8-blue)
![Static Badge](https://img.shields.io/badge/python-3.9-blue)
![Static Badge](https://img.shields.io/badge/python-3.10-blue)
![Static Badge](https://img.shields.io/badge/python-3.11-blue)

Dialogical Interpretability Tool for LLMs

## Running with conda / virtualenv
**Note: Please use Python 3.8+ and torch2.0+**
### Create the environment and install dependencies

#### Conda
```shell
conda create -n llmcheckup python=3.9
conda activate llmcheckup
```

#### venv
```shell
python -m venv venv
source venv/venv/activate
```

### 📝Install the requirements
```shell
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader averaged_perceptron_tagger
```

### 🚀Launch system
```shell
python flask_app.py
```

## Supported explainability methods
- Feature Attribution
  - Attention, Integrated gradient, etc.
  - Implemented by [inseq](https://github.com/inseq-team/inseq) package
- Semantic Similarity
- Free-text rationalization
  - Zero-shot CoT
  - Plan-and-Solve
  - Optimization by PROmpting (OPRO)
  - Any customized additional prompt according to users' wish
  - **_Notice_**: Above mentioned options can be freely chosen in the interface - "Prompt modification"
- Data Augmentation
  - Implemented by [NLPAug](https://github.com/makcedward/nlpaug) package or few-shot prompting
- Counterfactual Generation

## Models:
In our study, we identified three LLMs for our purposes.

- LLama2 (https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
  - Quantized Llama2 (https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ)
- Mistral (https://huggingface.co/mistralai/Mistral-7B-v0.1)
  - Quantized Mistral (https://huggingface.co/TheBloke/Mistral-7B-v0.1-GPTQ)
- Stable Beluga 2 (finetuned llama2-70B) (https://huggingface.co/petals-team/StableBeluga2)

### 🐳Deployment:
We support different methods for deployment:
- Original models
- Quantized by [GPTQ](https://arxiv.org/abs/2210.17323)
- Loading model in 8-bits by [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- peer2peer by [petals](https://github.com/bigscience-workshop/petals)

### ✏️Support:
|    Method    | Unix-based | Windows |
|:------------:|:----------:|:-------:|
|   Original   |    ✅    |   ✅   |
|     GPTQ     |    ✅     |   ✅   |
| Bitsanbytes* |    ✅    |   ✅   |
|   petals**   |    ✅     |   ❌  |

*: For Windows: if you encounter errors while installing bitsandbytes, then try: 
```bash
python -m pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
``` 

**: petals is currently not supported in windows, since a lot of Unix-specific things are used in petals. See issue [here](https://github.com/bigscience-workshop/petals/issues/488). petals is still usable if running ``LLMCheckup`` in Docker or WSL 2.

## Use case:
### Fact checking
Dataset: COVID-Fact

Link: https://github.com/asaakyan/covidfact

#### Structure
```
{
    Claim: ...,
    Evidence: ...,
    Label: ...,
}
```

### Commonsense Question Answering
Dataset: ECQA

Link: https://github.com/dair-iitd/ECQA-Dataset

```
{
    Question: ...,
    Multiple choices: ...,
    Correct answer: ...,
    Positive explanation: ...,
    Negative explanation: ...,
    Free-flow explanation: ...,
}
```

## Input with multi modalities
- Text
- Image
  - Image upload
  - Optical Character Recognition
    - Implemented by [EasyOCR](https://github.com/JaidedAI/EasyOCR) package
- Audio
  - A lightweight [fairseq s2t](https://huggingface.co/facebook/s2t-small-librispeech-asr) model from meta
  - If you encounter errors when reading recorded files: `soundfile.LibsndfileError: Error opening path_to_wav: Format not recognised.`. Then please try to install `ffmpeg`.
    - In Linux: ```sudo apt install ffmpeg``` or `pip3 install ffmpeg`
    - In Windows: Download `ffmpeg` from [here](https://github.com/BtbN/FFmpeg-Builds/releases) and add the path to system environment