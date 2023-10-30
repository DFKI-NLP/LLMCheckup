### Fine-tuning Models

To fine-tune a parsing model, run 

```shell
python parsing/t5/start_fine_tuning.py --gin parsing/t5/gin_configs/flan-t5-{base, large}.gin --dataset {boolq, daily_dialog, olid}
```

where `{small, base, large}` and `{boolq, daily_dialog, olid}` are one of the values in the set. Note, these experiments require use [Weights & Biases](https://wandb.ai/site) to track training and the best validation model.
