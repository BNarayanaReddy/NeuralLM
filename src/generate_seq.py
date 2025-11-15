import argparse
import torch
from utils import normalize_text
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.trainers import BpeTrainer
import os

# model classes for loading checkpoints
from model import Encoder, Decoder, Seq2Seq

def get_tokenizer(dataset, vocab_size, save_fldr, name = 'bpe'):
    if os.path.exists(save_fldr):
        save_path = os.path.join(save_fldr, 'tokenizer.json')
        return Tokenizer.from_file(save_path)
    if name == 'bpe':
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        tokenizer.train_from_iterator([dataset], trainer)
        if save_fldr:
            os.makedirs(save_fldr, exist_ok=True)
            tokenizer.save(os.path.join(save_fldr, 'tokenizer.json'))
            return tokenizer


def _load_tokenizer(path_or_dir):
    """Load a Tokenizer from a file or directory.
    If `path_or_dir` is a directory, expects tokenizer.json inside it.
    """
    if path_or_dir is None:
        raise ValueError("No tokenizer path provided")
    if os.path.isdir(path_or_dir):
        candidate = os.path.join(path_or_dir, 'tokenizer.json')
        if os.path.exists(candidate):
            return Tokenizer.from_file(candidate)
        raise FileNotFoundError(f"tokenizer.json not found in directory {path_or_dir}")
    if os.path.exists(path_or_dir):
        return Tokenizer.from_file(path_or_dir)
    raise FileNotFoundError(f"Tokenizer path {path_or_dir} does not exist")


def _build_model_from_checkpoint(ckpt_path, tokenizer, device, vocab_size=None):
    """Instantiate model using architecture constants and load weights from checkpoint.

    Tries common checkpoint dict layouts (state_dict, model_state_dict) and raw state_dict.
    """
    if vocab_size is None:
        # try to infer from tokenizer
        try:
            # Tokenizer has different APIs; try a few
            if hasattr(tokenizer, 'get_vocab_size'):
                vocab_size = tokenizer.get_vocab_size()
            elif hasattr(tokenizer, 'get_vocab'):
                vocab_size = len(tokenizer.get_vocab())
            else:
                vocab_size = VOCAB_SIZE
        except Exception:
            vocab_size = VOCAB_SIZE

    # instantiate encoder/decoder/seq2seq using file-level constants
    encoder = Encoder(vocab_size, EMB_DIM, ENC_HIDDEN_DIM, DROPOUT)
    decoder = Decoder(vocab_size, EMB_DIM, ENC_HIDDEN_DIM, DEC_HIDDEN_DIM, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    if not ckpt_path:
        raise ValueError("No checkpoint path provided to load model")

    ckpt = torch.load(ckpt_path, map_location=device)

    # heuristics to find state dict
    state = None
    if isinstance(ckpt, dict):
        # common keys
        for key in ('model_state_dict', 'state_dict', 'model'):
            if key in ckpt:
                state = ckpt[key]
                break
        if state is None:
            # maybe the dict itself is the state dict
            state = ckpt
    else:
        state = ckpt

    try:
        model.load_state_dict(state)
    except Exception:
        # last resort: try if state contains nested 'model_state_dict'
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            raise

    model.to(device)
    model.eval()
    return model

seed = 1234
DATA_DIR = 'data'
SEQ_LEN = 50
VOCAB_SIZE = 5000
EMB_DIM = 256
ENC_HIDDEN_DIM = 512
DEC_HIDDEN_DIM = 512
DROPOUT = 0.3
N_EPOCHS = 200
LEARNING_RATE = 1e-2
BATCH_SIZE = 64
# Output directories
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'runs'
tokenizer_path = "tknzer_dir"
special_tokens = ["<pad>", "<st>", "<end>", "<unk>"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_seq(model, texts, tokenizer, seq_len, special_ids, gen_len=100):
    model.eval()
    samples = []
    pad, st, end, full_stop, quest, excl= special_ids

    with torch.no_grad():
        for text in texts:
            # prepare encoder input
            text = normalize_text(text)
            ids = tokenizer.encode(text).ids
            if len(ids) < seq_len:
                ids = ids + [pad] * (seq_len - len(ids))
            else:
                print("Warning: More than seq length -- considering first 50 tokens:")
                ids = ids[:seq_len]

            src = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            # print(src.shape)
            # print(src)

            # start decoder with only the start token (length = 1)
            dec_in = torch.tensor([[st]], dtype=torch.long, device=DEVICE)

            generated = []
            for step in range(gen_len):
                out = model(src, dec_in)          # [1, cur_dec_len, vocab_size]
                next_token = int(out[0, -1].argmax().cpu().item())  # last timestep prediction
                generated.append(next_token)
                
                # if next_token == end:
                #     break

                

                # append predicted token to decoder input for next step
                dec_in = torch.cat(
                    [dec_in, torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)],
                    dim=1
                )
            # convert ids -> tokens
            # print("Out Tokens", generated)
            tokens = [tokenizer.id_to_token(tid) for tid in generated]
            samples.append((text, tokens))

    print("=== Sample generations ===")
    for idx, (inp, toks) in enumerate(samples, 1):
        print(f"[{idx}] INPUT : {inp}")
        print(f"OUTPUT: {' '.join(toks)}")
    print("======================================================")
    return samples


def _get_special_ids_from_tokenizer(tokenizer):
    """Return a list of six special ids used by generate_seq: pad, st, end, full_stop, quest, excl.

    Falls back to <unk> or 0 when a token cannot be resolved.
    """
    desired = ["<pad>", "<st>", "<end>", ".", "?", "!"]
    ids = []
    # prefer tokenizer.token_to_id, fall back to special_tokens mapping when available
    for tok in desired:
        tid = None
        try:
            if hasattr(tokenizer, 'token_to_id'):
                tid = tokenizer.token_to_id(tok)
            elif hasattr(tokenizer, 'get_vocab'):
                vocab = tokenizer.get_vocab()
                tid = vocab.get(tok)
        except Exception:
            tid = None

        if tid is None:
            # try to map known special tokens
            if tok in special_tokens:
                try:
                    tid = tokenizer.token_to_id(tok)
                except Exception:
                    tid = None
        if tid is None:
            # fallback to unk or 0
            try:
                tid = tokenizer.token_to_id('<unk>') or 0
            except Exception:
                tid = 0
        ids.append(tid)
    return ids


def main():
    parser = argparse.ArgumentParser(description='Generate sequences with a trained Seq2Seq model')
    parser.add_argument('-c', '--checkpoint', help='Path to model checkpoint (.pt/.pth)', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--text', help='Input text (can be repeated)', action='append')
    group.add_argument('-i', '--input-file', help='Path to a file with one input per line')
    parser.add_argument('-k', '--tokenizer', help='Tokenizer file or directory containing tokenizer.json', default=tokenizer_path)
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN, help='Sequence length to use for encoder (default from file)')
    parser.add_argument('--gen-len', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--vocab-size', type=int, default=None, help='Vocab size (optional, inferred from tokenizer if not provided)')
    args = parser.parse_args()

    device = torch.device(args.device)

    # load tokenizer
    tokenizer = _load_tokenizer(args.tokenizer)

    # prepare texts
    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(args.input_file)
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = args.text or []

    # load model
    model = _build_model_from_checkpoint(args.checkpoint, tokenizer, device, vocab_size=args.vocab_size)

    special_ids = _get_special_ids_from_tokenizer(tokenizer)

    # call generation
    generate_seq(model, texts, tokenizer, seq_len=args.seq_len, special_ids=special_ids, gen_len=args.gen_len)


if __name__ == '__main__':
    main()


