# NeuralLM 
- `In Progress`
## Training Scripts on Kaggle

The complete training notebooks for each configuration are available on Kaggle:

| Configuration | Description | Link |
|--------------|-------------|------|
| **Config 1** | Best NLM training setup | https://www.kaggle.com/code/bnarayanareddy/iiith-neurallangmodel-1 |
| **Config 2** | Modified architecture / hyperparameters | https://www.kaggle.com/code/bnarayanareddy/iiith-neurallangmodel-3 |
| **Config 3** | Increased input sequence length | https://www.kaggle.com/code/bnarayanareddy/iiith-neurallangmodel-2 |

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