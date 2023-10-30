# Directory for fine-tuned parsing models

To test out the current model, download it from the link...

### FLAN-T5-base (250M)

| Dataset    | Link                                                                            | Val Accuracy |
|------------|---------------------------------------------------------------------------------|--------------|
| **BoolQ**  | Due to anonymization not shown any links | 95 %         |
| **DA**     | Due to anonymization not shown any links | 96 %         |
| **OLID**   | Due to anonymization not shown any links | 93 %         |

...and put the folder (flan-t5-base) inside parsing/t5/models (this directory).

Afterwards, change `GlobalArgs.config` in *global_config.gin* in the root directory to the following:

```
GlobalArgs.config = "./configs/boolq_flan-t5.gin"
```
