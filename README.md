# NeuralLM 
## Training Scripts on Kaggle

The complete training notebooks for each configuration are available on Kaggle:

| Configuration | Description |
|--------------|-------------|
| [**Config 1**](https://www.kaggle.com/code/bnarayanareddy/iiith-neurallangmodel-1) | Best NLM training setup |
| [**Config 2**](https://www.kaggle.com/code/bnarayanareddy/iiith-neurallangmodel-2) | Increased input sequence length |
| [**Config 3**](https://www.kaggle.com/code/bnarayanareddy/iiith-neurallangmodel-3) | Modified architecture / hyperparameters |


---


## Inference

* Download checkpoint from the releases tab and pass the path in place of checkpoint.pt

```bash
    cd src/
    python generate_seq.py -c checkpoint.pt -t "Hi complete this sequence" -k "../tokenizer"
```

## Plots

```bash
    tensorboard --logdir events/config1
```