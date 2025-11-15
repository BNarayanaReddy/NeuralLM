import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ENCODER
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.enc_hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
    
    def forward(self, x):
        # x: [batch, seq_len]
        emb = self.embedding(x)  # [batch, seq_len, emb_dim]
        outputs, (h, c) = self.lstm(emb)  # outputs: [batch, seq_len, 2*hidden]
        return outputs, (h, c)

# ATTENTION 
class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        
        # We need to align the dimensions of encoder outputs and decoder hidden state
        # enc_hidden_dim is for one direction, but encoder is bidirectional (2*)
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias = False)
        
    def forward(self, dec_hidden, enc_outputs):
        # dec_hidden: [batch, dec_hidden_dim] (from the *top layer* of decoder)
        # enc_outputs: [batch, src_len, enc_hidden_dim * 2]
        
        batch_size = enc_outputs.shape[0]
        src_len = enc_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times to concatenate
        # dec_hidden: [batch, src_len, dec_hidden_dim]
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # energy: [batch, src_len, (enc_hidden * 2) + dec_hidden]
        energy_input = torch.cat((dec_hidden, enc_outputs), dim = 2)
        
        # energy: [batch, src_len, dec_hidden_dim]
        energy = torch.tanh(self.attn(energy_input))
        
        # v(energy): [batch, src_len, 1] -> [batch, src_len]
        attention = self.v(energy).squeeze(2)
        
        # Return softmax'd weights
        return F.softmax(attention, dim=1)

# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout=0.1):
        super().__init__()
        self.dec_hidden_dim = dec_hidden_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        self.attention = Attention(enc_hidden_dim, dec_hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=emb_dim + (enc_hidden_dim * 2),
            hidden_size=dec_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(dec_hidden_dim + (enc_hidden_dim * 2), vocab_size)
    
    def forward(self, dec_input, dec_hidden, enc_outputs):
        # dec_input: [batch] -> current token IDs
        # dec_hidden: (h, c) from previous step
        # enc_outputs: [batch, src_len, 2*enc_hidden_dim]
        
        # dec_input [batch] -> [batch, 1]
        dec_input = dec_input.unsqueeze(1)
        
        # emb: [batch, 1, emb_dim]
        emb = self.embedding(dec_input)
        
       
        a = self.attention(dec_hidden[0][-1], enc_outputs)
        
        a = a.unsqueeze(1)
        
        context = torch.bmm(a, enc_outputs)
        
        lstm_input = torch.cat([emb, context], dim=2)
        output, dec_hidden = self.lstm(lstm_input, dec_hidden)
        
        output = output.squeeze(1)
        context = context.squeeze(1)

        concat_output = torch.cat([output, context], dim=1)
        
        logits = self.fc(concat_output)
        
        return logits, dec_hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        self.enc_num_layers = self.encoder.lstm.num_layers
        self.enc_num_directions = 2 if self.encoder.lstm.bidirectional else 1
        self.enc_hidden_dim = self.encoder.enc_hidden_dim
        self.dec_hidden_dim = self.decoder.dec_hidden_dim
        
        self.fc_hidden = nn.Linear(self.enc_hidden_dim * self.enc_num_directions, self.dec_hidden_dim)
        self.fc_cell = nn.Linear(self.enc_hidden_dim * self.enc_num_directions, self.dec_hidden_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        enc_outputs, (h, c) = self.encoder(src)
        
        h = h.view(self.enc_num_layers, self.enc_num_directions, batch_size, self.enc_hidden_dim)
        c = c.view(self.enc_num_layers, self.enc_num_directions, batch_size, self.enc_hidden_dim)
        
        h_cat = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=2)
        c_cat = torch.cat((c[:, 0, :, :], c[:, 1, :, :]), dim=2)
        
        dec_hidden = (torch.tanh(self.fc_hidden(h_cat)), 
                      torch.tanh(self.fc_cell(c_cat)))
        
        dec_input = trg[:, 0]
        
        # Loop from 0, store at t
        for t in range(trg_len):
            
            # The decoder's forward pass now includes the attention mechanism
            logits, dec_hidden = self.decoder(dec_input, dec_hidden, enc_outputs)
            
            outputs[:, t] = logits
            
            use_teacher_force = random.random() < teacher_forcing_ratio
            
            if use_teacher_force:
                if t < trg_len - 1:
                    dec_input = trg[:, t+1]
                else:
                    break
            else:
                top1 = logits.argmax(1)
                dec_input = top1
            
        return outputs