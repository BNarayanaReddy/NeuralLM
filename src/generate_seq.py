import argparse
import torch
from utils import normalize_text
from tokenizers import Tokenizer
import os
from config import Config
# model classes for loading checkpoints
from model import Encoder, Decoder, Seq2Seq


def get_tokenizer(save_fldr, name = 'bpe'):
    if os.path.exists(save_fldr):
        save_path = os.path.join(save_fldr, 'tokenizer.json')
        return Tokenizer.from_file(save_path)
    else:
        print("Tokenizer not Found")


def build_model(config, ckpt_path, tokenizer_dir):
    tokenizer = get_tokenizer(tokenizer_dir)
    # instantiate encoder/decoder/seq2seq using file-level constants
    encoder = Encoder(config.VOCAB_SIZE, config.EMB_DIM, config.ENC_HIDDEN_DIM, config.DROPOUT)
    decoder = Decoder(config.VOCAB_SIZE, config.EMB_DIM, config.ENC_HIDDEN_DIM, config.DEC_HIDDEN_DIM, config.DROPOUT)
    model = Seq2Seq(encoder, decoder, config.DEVICE).to(config.DEVICE)

    if not ckpt_path:
        raise ValueError("No checkpoint path provided to load model")

    ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
    model.load_state_dict(ckpt)
    
    return model, tokenizer

def generate_seq(model, texts, tokenizer, config, gen_len=100):
    model.eval()
    seq_len = config.SEQ_LEN
    device = config.DEVICE
    special_tokens = config.special_tokens
    samples = []
    pad, st, end, *_ = [tokenizer.token_to_id(i) for i in special_tokens]

    with torch.no_grad():
        if type(texts) == str:
            texts = [texts]
        for text in texts:
            # prepare encoder input
            text = normalize_text(text)
            ids = tokenizer.encode(text).ids
            if len(ids) < seq_len:
                ids = ids + [pad] * (seq_len - len(ids))
            else:
                print("Warning: More than seq length -- considering first 50 tokens:")
                ids = ids[:seq_len]

            src = torch.tensor([ids], dtype=torch.long, device=device)
            dec_in = torch.tensor([[st]], dtype=torch.long, device=device)

            generated = []
            for step in range(gen_len):
                out = model(src, dec_in)          # [1, cur_dec_len, vocab_size]
                next_token = int(out[0, -1].argmax().cpu().item())  # last timestep prediction
                
                
                if next_token == end:
                    break
                generated.append(next_token)

                # append predicted token to decoder input for next step
                dec_in = torch.cat(
                    [dec_in, torch.tensor([[next_token]], dtype=torch.long, device=device)],
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


def main():
    parser = argparse.ArgumentParser(description='Generate sequences with a trained Seq2Seq model')
    parser.add_argument('-c', '--checkpoint', help='Path to model checkpoint (.pt)', required=True)
    parser.add_argument('-t', '--text', help='Input text (can be repeated), str or list of text seq', action='append')
    parser.add_argument('-k', '--tokenizer_dir', help='Tokenizer directory containing tokenizer.json')
    parser.add_argument('--config', choices=(1, 2, 3), default=1, type=int, help="Type of config to try")
    parser.add_argument('--gen-len', type=int, default=100, help='Number of tokens to generate')
    args = parser.parse_args()
    config = Config(args.config)
    model, tknzr = build_model(config, args.checkpoint, args.tokenizer_dir)
    generate_seq(model, args.text, tknzr, config, args.gen_len)

if __name__ == '__main__':
    main()


