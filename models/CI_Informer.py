import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding, CI_embedding, DataEmbedding_CI, positional_encoding

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
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.d_model = configs.d_model
        self.nvars = configs.enc_in
        
        self.output_attention = configs.output_attention
        self.subtract_last = configs.subtract_last
        self.encoder_decoder = configs.encoder_decoder

        self.embedding = CI_embedding(configs.d_model)
        self.W_pos = positional_encoding(pe = 'zeros', learn_pe=True, q_len = self.seq_len, d_model = self.d_model)
        
        # Embedding
        
        self.enc_embedding = DataEmbedding_CI(1, configs.d_model, self.nvars, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding_CI(1, configs.d_model, self.nvars, configs.embed, configs.freq,
                                           configs.dropout)

        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        # self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)

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
            # projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        if self.encoder_decoder:
            self.head = Flatten_Head(individual = False, 
                                    n_vars = self.nvars, nf = self.d_model*(self.label_len+self.pred_len), 
                                    target_window = self.pred_len, head_dropout=0)
        else:
            self.head = Flatten_Head(individual = False, 
                                    n_vars = self.nvars, nf = self.d_model*self.seq_len, 
                                    target_window = self.pred_len, head_dropout=0)    
        

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None): 

        '''
        x_enc: (bsz, seq_len, nvars)
        x_mark_enc: (bsz, seq_len, 4)
        x_dec: (bsz, label_len+pred_len, nvars)
        x_mark_dec: (bsz, label_len+pred_len, 4)
        '''

        bsz, seq_len, nvars = x_enc.shape

        if self.subtract_last:
            seq_last = x_enc[:, -1:, :].detach()
            x_enc = x_enc - seq_last

        if self.encoder_decoder:

            x_enc = x_enc.reshape(bsz*nvars, seq_len, 1)
            x_dec = x_dec.reshape(bsz*nvars, self.label_len+self.pred_len, 1)
            x_mark_enc = x_mark_enc.reshape(bsz, seq_len, 4)
            x_mark_dec = x_mark_dec.reshape(bsz, self.label_len+self.pred_len, 4)
            
            enc_out = self.enc_embedding(x_enc, x_mark_enc) # (bsz, seq_len, d_model)
            enc_out = enc_out.reshape(bsz*nvars, seq_len, self.d_model)
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
            enc_out = enc_out.reshape(-1, self.seq_len, self.d_model)

            dec_out = self.dec_embedding(x_dec, x_mark_dec) # (bsz, seq_len, d_model)
            dec_out = dec_out.reshape(-1, self.label_len+self.pred_len, self.d_model)
            output = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask) # (bsz * nvars, label_len+pred_len, d_model)
            
            dec_out = dec_out.reshape(-1, self.nvars, self.label_len+self.pred_len, self.d_model) # (bsz, nvars, label_len+pred_len, d_model)
            
            output = self.head(dec_out) # (bsz, nvars, label_len+pred_len)
            output = output.permute(0,2,1) # (bsz, label_len+pred_len, nvars)
        
        else:

            output = self.embedding(x_enc) # (bsz, seq_len, nvars, d_model)
            output = output.permute(0,2,1,3) # (bsz, nvars, seq_len, d_model)
            output = output.reshape(-1, self.seq_len, self.d_model) # (bsz*nvars, seq_len, d_model)
            output = output + self.W_pos # (bsz*nvars, seq_len, d_model)

            output, _ = self.encoder(output, attn_mask = None)
            output = output.reshape(-1, self.nvars, self.seq_len, self.d_model) # (bsz, nvars, seq_len, d_model) # Connect this output to Flatten Head for forecasting
            
            output = self.head(output) # (bsz, nvars, pred_len)
            output = output.permute(0,2,1) # (bsz, pred_len, nvars)

        if self.subtract_last: 
            output = output + seq_last

        if self.output_attention:
            return output, attns
        else:
            return output  # [B, L, D]