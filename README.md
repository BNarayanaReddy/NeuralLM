# NeuralLM

A sequence-to-sequence neural language model implementation with encoder-decoder architecture using LSTM networks and attention mechanisms.

## Overview

This project implements a neural language model for text generation using:

- **Encoder**: Bidirectional LSTM that processes input sequences
- **Decoder**: LSTM-based decoder with attention mechanism
- **Tokenizer**: BPE (Byte Pair Encoding) tokenizer for text processing

## Project Structure

```txt
NeuralLM/
├── src/                          # Source code
│   ├── model.py                 # Encoder, Decoder, and Seq2Seq model definitions
│   ├── config.py                # Configuration classes for different setups
│   ├── generate_seq.py          # Inference script for sequence generation
│   └── utils.py                 # Utility functions
├── train/
│   ├── Pride_and_Prejudice-Jane_Austen.txt  # Training data
│   └── notebook/                # Jupyter notebooks for training
│       ├── iiith-neurallangmodel-1.ipynb
│       ├── iiith-neurallangmodel-2.ipynb
│       └── iiith-neurallangmodel-3.ipynb
├── tokenizer/                   # Pre-trained BPE tokenizer
│   └── tokenizer.json
├── events/                       # TensorBoard logs
│   ├── config1/
│   ├── config2/
│   └── config3/
└── requirements.txt             # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Three different configurations are available (in `src/config.py`):

| Config | SEQ_LEN | EMB_DIM | HIDDEN_DIM | Description |
|--------|---------|---------|-----------|-------------|
| **1** (default) | 50 | 256 | 512 | Standard setup - Best performance |
| **2** | 100 | 256 | 512 | Extended sequence length |
| **3** | 50 | 128 | 256 | Reduced model size |


## Training

Training notebooks are available in `train/notebook/`:

- **Config 1**: `iiith-neurallangmodel-1.ipynb` - Best NLM training setup
- **Config 2**: `iiith-neurallangmodel-2.ipynb` - Extended sequence length
- **Config 3**: `iiith-neurallangmodel-3.ipynb` - Lighter model variant

## Inference

To generate text sequences using a trained model:

```bash
cd src/
python generate_seq.py -c <checkpoint.pt> -t "Your input text here" -k "../tokenizer"
```

**Arguments**

- `-c, --checkpoint` **(required)**: Path to model checkpoint file (.pt)
- `-t, --text` **(required)**: Input text prompt (can be repeated for multiple prompts)
- `-k, --tokenizer_dir`: Directory containing `tokenizer.json`
- `--config`: Configuration version to use (1, 2, or 3; default: 1)
- `--gen-len`: Number of tokens to generate (default: 100)

**Example**

```bash
python generate_seq.py -c ../checkpoints/model_config1.pt -t "It was a dark" -k "../tokenizer" --gen-len 50
```


## Visualization

View training logs with TensorBoard:

```bash
tensorboard --logdir events/config1
```

or for other configurations:

```bash
tensorboard --logdir events/config2
tensorboard --logdir events/config3
```