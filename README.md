# NeuralLM 
- `In Progress`
## Training Scripts on Kaggle

The complete training notebooks for each configuration are available on Kaggle:

| Configuration | Description |
|--------------|-------------|
| [**Config 1**](https://www.kaggle.com/code/bnarayanareddy/iiith-neurallangmodel-1) | Best NLM training setup |
| [**Config 2**](https://www.kaggle.com/code/bnarayanareddy/iiith-neurallangmodel-3 ) | Modified architecture / hyperparameters |
| [**Config 3**](https://www.kaggle.com/code/bnarayanareddy/iiith-neurallangmodel-2) | Increased input sequence length |

---

## Plots

```bash
    tensorboard --logdir events/config1
    tensorboard --logdir events/config2
    tensorboard --logdir events/config3
```

## Inference
```bash
    cd src/
    python /home/narayana/Projects/iiit_ssmt2/LangModel/NeuralLM/NeuralLM/src/generate_seq.py
```
