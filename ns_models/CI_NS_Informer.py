import torch
import torch.nn as nn
from ns_layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from ns_layers.SelfAttention_Family import DSProbAttention, AttentionLayer
from layers.Embed import CI_embedding, DataEmbedding, positional_encoding, DataEmbedding, DataEmbedding_CI


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)          # B x 1 x E
        x = torch.cat([x, stats], dim=1) # B x 2 x E
        x = x.view(batch_size, -1) # B x 2E
        y = self.backbone(x)       # B x O

        return y

class Model(nn.Module):
    """
    Non-stationary Informer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.d_model = configs.d_model
        self.nvars = configs.enc_in

        self.encoder_decoder = configs.encoder_decoder
        self.output_attention = configs.output_attention
        self.subtract_last = configs.subtract_last

        self.embedding = CI_embedding(self.d_model)
        self.W_pos = positional_encoding(pe = 'zeros', learn_pe=True, q_len = self.seq_len, d_model = self.d_model)
        
        # Embedding
        self.enc_embedding = DataEmbedding_CI(1, configs.d_model, self.nvars, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding_CI(1, configs.d_model, self.nvars, configs.embed, configs.freq,
                                           configs.dropout)

        
        self.tau_learner = Projector(
                                enc_in = self.nvars, 
                                seq_len = self.seq_len, 
                                hidden_dims= configs.p_hidden_dims, 
                                hidden_layers = configs.p_hidden_layers, 
                                output_dim=1 * self.nvars, 
                                kernel_size=3
                                )
        
        self.delta_learner = Projector(
                                enc_in = self.nvars, 
                                seq_len = self.seq_len, 
                                hidden_dims= configs.p_hidden_dims, 
                                hidden_layers = configs.p_hidden_layers, 
                                output_dim=self.seq_len * self.nvars, 
                                kernel_size=3
                                )
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSProbAttention(False, configs.factor, attention_dropout=configs.dropout,
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
                        DSProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
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

        bsz, seq_len, nvars = x_enc.shape

        x_raw = x_enc.clone().detach()

        if self.subtract_last:
            seq_last = x_raw[:, -1:, :].detach()
            x_enc = x_enc - seq_last

        x_mean = x_enc.mean(1, keepdim=True).detach() # bsz x 1 x nvars
        x_enc = x_enc - x_mean # bsz x seq_len x nvars

        x_std = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # bsz x 1 x nvars
        x_enc = x_enc / x_std # bsz x seq_len x nvars

        tau = self.tau_learner(x_raw, x_std).exp().reshape(-1,1)   # bsz, seq_len, nvars & bsz, 1, nvars -> bsz, nvars
        delta = self.delta_learner(x_raw, x_mean).reshape(bsz*nvars, seq_len)   # bsz, seq_len, nvars & bsz, 1, nvars -> bsz, nvars * seq_len
    
        if self.encoder_decoder:
            x_enc = x_enc.reshape(bsz*nvars, self.seq_len, 1)
            x_dec = x_dec.reshape(bsz*nvars, self.label_len+self.pred_len, 1)
            x_mark_enc = x_mark_enc.reshape(bsz, self.seq_len, 4)
            x_mark_dec = x_mark_dec.reshape(bsz, self.label_len+self.pred_len, 4)

            enc_out = self.enc_embedding(x_enc, x_mark_enc) # (bsz, seq_len, d_model)
            enc_out = enc_out.reshape(bsz*nvars, self.seq_len, self.d_model)
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)
            enc_out = enc_out.reshape(-1, self.seq_len, self.d_model)

            dec_out = self.dec_embedding(x_dec, x_mark_dec) # (bsz, seq_len, d_model)
            dec_out = dec_out.reshape(-1, self.label_len+self.pred_len, self.d_model)
            output = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta) # (bsz * nvars, label_len+pred_len, d_model)
            
            dec_out = dec_out.reshape(-1, self.nvars, self.label_len+self.pred_len, self.d_model) # (bsz, nvars, label_len+pred_len, d_model)
            
            output = self.head(dec_out) # (bsz, nvars, pred_len)
            output = output.permute(0,2,1)

        else:
            output = self.embedding(x_enc) # (bsz, seq_len, nvars, d_model)
            output = output.permute(0,2,1,3) # (bsz, nvars, seq_len, d_model)
            output = output.reshape(-1, self.seq_len, self.d_model) # (bsz*nvars, seq_len, d_model)
            output = output + self.W_pos # (bsz*nvars, seq_len, d_model)

            output, _ = self.encoder(output, attn_mask = enc_self_mask, tau = tau, delta = delta)
            output = output.reshape(-1, self.nvars, self.seq_len, self.d_model) # (bsz, nvars, seq_len, d_model) # Connect this output to Flatten Head for forecasting
            
            output = self.head(output) # (bsz, nvars, pred_len)
            output = output.permute(0,2,1) # (bsz, pred_len, nvars)

        output = output * x_std + x_mean # bsz x pred_len x nvars

        if self.subtract_last: 
            output = output + seq_last

        return output

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