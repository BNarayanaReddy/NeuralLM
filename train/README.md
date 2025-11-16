# Neural Language Model Training Notebooks

This directory contains three training notebooks that implement Sequence-to-Sequence (Seq2Seq) neural language models with different hyperparameter configurations. All notebooks train on `Pride_and_Prejudice-Jane_Austen.txt`

## Overview

The three notebooks explore different model configurations to evaluate the trade-offs between model capacity, training time, and generalization performance:

- **iiith-neurallangmodel-1.ipynb** (Config 1)
- **iiith-neurallangmodel-2.ipynb** (Config 2) 
- **iiith-neurallangmodel-3.ipynb** (Config 3)

All models implement a Seq2Seq architecture with:
- **Encoder**: Bidirectional LSTM with attention
- **Decoder**: LSTM decoder with Bahdanau attention mechanism
- **Task**: Next-token prediction (language modeling)

---

## Configuration Comparison

### Model Hyperparameters

| Parameter | Config 1 (Large) | Config 2 (Medium) | Config 3 (Small) |
|-----------|-----------------|------------------|-----------------|
| **SEQ_LEN** | 50 | 100 | 50 |
| **VOCAB_SIZE** | 5,000 | 5,000 | 5,000 |
| **EMB_DIM** | 256 | 256 | 128 |
| **ENC_HIDDEN_DIM** | 512 | 512 | 256 |
| **DEC_HIDDEN_DIM** | 512 | 512 | 256 |
| **LSTM Layers (Encoder)** | 2 | 2 | 4 |
| **LSTM Layers (Decoder)** | 2 | 2 | 4 |

### Training Hyperparameters

| Parameter | Config 1 | Config 2 | Config 3 |
|-----------|----------|----------|----------|
| **N_EPOCHS** | 200 | 200 | 200 |
| **LEARNING_RATE** | 1e-2 | 1e-2 | 1e-2 |
| **BATCH_SIZE** | 64 | 64 | 64 |
| **LR Scheduler** | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau |
| **Teacher Forcing** | Random (0.5) | Random (0.5) | Always (1.0) |

---

## Architecture Details

### Model Components

#### 1. **Encoder (Bidirectional LSTM)**
```
Input (batch, seq_len) 
	- Embedding (batch, seq_len, emb_dim)
	- Bi-LSTM (batch, seq_len, 2*hidden_dim)
	- Outputs + Hidden States
```

Key characteristics:
- Bidirectional processing captures context from both directions
- Multiple layers allow hierarchical feature extraction
- Config 1&2 use 2 layers with larger hidden dimensions (512)
- Config 3 uses 4 layers with smaller hidden dimensions (256)

#### 2. **Attention Mechanism**
```
The attention mechanism allows the decoder to focus on relevant parts of the input sequence dynamically.
```

#### 3. **Decoder (LSTM)**
```
Start Token + Attention Context
	- LSTM Cell
	- Output + New Hidden States
	- Linear Projection -> Vocabulary
```

Key characteristics:
- Uses attention to incorporate encoder information
- Teacher forcing during training (with varying ratios)
- Autoregressive decoding during inference

---


## Dataset & Preprocessing

### Text Source
- **File**: `Pride_and_Prejudice-Jane_Austen.txt`
- **Preprocessing**:
	- Lowercase conversion
	- Unicode normalization (NFKC)
	- Whitespace normalization
	- Special character removal
	- Abbreviation handling (mr., mrs., etc.)

### Tokenization
- **Method**: Byte-Pair Encoding (BPE)
- **Vocab Size**: 5,000
- **Special Tokens**: `<pad>`, `<st>` (start), `<end>`, and standard punctuation

### Data Split
- **Train**: 80% of tokenized data
- **Validation**: 20% of tokenized data
- **Sequence Creation**: Sentence-aware (sequences end at sentence boundaries)

---

### Monitoring
- **TensorBoard Logging**: Tracks loss, perplexity, and learning rate
- **Validation Frequency**: Every epoch
- **Sample Generation**: Every 5 epochs during training

---
