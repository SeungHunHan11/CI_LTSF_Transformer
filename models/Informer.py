import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x

class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.subtract_last = configs.subtract_last
        self.encoder_decoder = configs.encoder_decoder

        self.head = Flatten_Head(individual = False, 
                                 n_vars = configs.c_in, nf = configs.d_model*configs.seq_len, 
                                 target_window = configs.pred_len, head_dropout=0)     
        
        
        # Embedding

        # if configs.CI:
            # self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq,
            #                            configs.dropout)

            # self.dec_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq,
            #                                    configs.dropout)


        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None): 

        '''
        x_enc: (bsz, seq_len, nvars)
        x_mark_enc: (bsz, seq_len, 4)
        x_dec: (bsz, label_len+pred_len, nvars)
        x_mark_dec: (bsz, label_len+pred_len, 4)
        '''

        # bsz, seq_len, nvars = x_enc.shape

        if self.subtract_last:
            seq_last = x_enc[:, -1:, :].detach()
            x_enc = x_enc - seq_last

        # if self.CI:

            # x_enc = x_enc.unsqueeze(-1).reshape(bsz, nvars, seq_len, 1)
            # x_dec = x_dec.unsqueeze(-1).reshape(bsz, nvars, label_len+pred_len, 1)
            # x_mark_enc = x_mark_enc.unsqueeze(-1).reshape(bsz, 4, seq_len, 1)
            # x_mark_dec = x_mark_dec.unsqueeze(-1).reshape(bsz, 4, label_len+pred_len, 1)

            # enc_out = self.enc_embedding(x_enc, x_mark_enc) # (bsz, seq_len, d_model)
            # enc_out = enc_out.reshape(-1, self.seq_len, self.d_model)

            # dec_out = self.DataEmbedding(x_dec, x_mark_dec) # (bsz, seq_len, d_model)
            # dec_out = dec_out.reshape(-1, self.label_len+self.pred_len, self.d_model)
            # dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask) # (bsz * nvars, label_len+pred_len, d_model)
            # dec_out = self.head(dec_out) # (bsz, nvars, label_len+pred_len)

        # else:

        enc_out = self.enc_embedding(x_enc, x_mark_enc) # (bsz, seq_len, d_model)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) # (bsz, seq_len, d_model)

        dec_out = self.dec_embedding(x_dec, x_mark_dec) # (bsz, label_len+pred_len, d_model)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask) # (bsz, label_len+pred_len, d_model)

        if self.subtract_last: 
            dec_out = dec_out + seq_last

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
