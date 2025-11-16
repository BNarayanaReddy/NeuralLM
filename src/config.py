import torch



# get config
class Config():
    def __init__(self, type):
        self.num = type
        self.seed = 1234
        self.DATA_DIR = 'data'
        self.SEQ_LEN = 50
        self.VOCAB_SIZE = 5000
        self.EMB_DIM = 256
        self.ENC_HIDDEN_DIM = 512
        self.DEC_HIDDEN_DIM = 512
        self.DROPOUT = 0.3
        self.N_EPOCHS = 200
        self.LEARNING_RATE = 1e-2
        self.BATCH_SIZE = 64
        # Output directories
        self.CHECKPOINT_DIR = 'checkpoints'
        self.LOG_DIR = 'runs'
        self.tokenizer_path = "tknzer_dir"
        self.special_tokens = ["<pad>", "<st>", "<end>", "<unk>"]
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.num == 2:
            self.EMB_DIM = 128
            self.ENC_HIDDEN_DIM = 256
            self.DEC_HIDDEN_DIM = 256
        if self.num == 3:
            self.SEQ_LEN = 100
    