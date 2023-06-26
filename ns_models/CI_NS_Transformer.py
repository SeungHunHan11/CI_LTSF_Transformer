import torch.nn as nn
import torch.nn.functional as F
from ns_layers.Embed import CI_embedding, positional_encoding
from ns_layers.SelfAttention_Family import DSAttention, AttentionLayer
from ns_layers.Transformer_EncDec import Encoder, EncoderLayer

import torch

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
    Non-Stationary Transformer (Encoder) with Channel-Independece
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.nvars = configs.enc_in
        self.subtract_last = configs.subtract_last

        self.embedding = CI_embedding(self.d_model)
        
        self.W_pos = positional_encoding(pe = 'zeros', learn_pe=True, q_len = self.seq_len, d_model = self.d_model)

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
        
        self.encoder = Encoder(
                                [
                                    EncoderLayer(
                                        AttentionLayer(
                                            DSAttention(False, 5, attention_dropout=0.1,
                                                        output_attention=False), self.d_model, configs.n_heads),
                                                        configs.d_model,
                                                        configs.d_ff,
                                                        dropout=configs.dropout,
                                                        activation=configs.activation
                                    ) for l in range(configs.e_layers
                                                     )
                                ],
                                norm_layer=torch.nn.LayerNorm(configs.d_model)
                            )           

        self.head = Flatten_Head(individual = False, 
                                 n_vars = self.nvars, nf = self.d_model*self.seq_len, 
                                 target_window = self.pred_len, head_dropout=0)            


    def forward(self, x): # (bsz, seq_len, nvars)

        bsz, seq_len, nvars = x.shape

        x_raw = x.clone().detach()
        
        if self.subtract_last:
            seq_last = x_raw[:, -1:, :].detach()
            x = x - seq_last
        
        x_mean = x.mean(1, keepdim=True).detach() # bsz x 1 x nvars
        x = x - x_mean # bsz x seq_len x nvars

        x_std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # bsz x 1 x nvars

        x = x / x_std # bsz x seq_len x nvars

        tau = self.tau_learner(x_raw, x_std).exp().reshape(-1,1)   # bsz, seq_len, nvars & bsz, 1, nvars -> bsz, nvars
        delta = self.delta_learner(x_raw, x_mean).reshape(bsz*nvars, seq_len)   # bsz, seq_len, nvars & bsz, 1, nvars -> bsz, nvars * seq_len
    
        output = self.embedding(x) # (bsz, seq_len, nvars, d_model)
        output = output.permute(0,2,1,3) # (bsz, nvars, seq_len, d_model)
        output = output.reshape(-1, self.seq_len, self.d_model) # (bsz*nvars, seq_len, d_model)
        output = output + self.W_pos # (bsz*nvars, seq_len, d_model)

        output, _ = self.encoder(output, attn_mask = None, tau = tau, delta = delta)
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
    




# from data_provider.data_loader import Dataset_ETT_hour

# ds = Dataset_ETT_hour('/directory/Nonstationary_Transformers/CI_NS_Transformer/dataset/ETT-small',
#                  flag='train',
#                  size = None,
#                  features='MS',
#                  target='OT')

# bsz = 1
# nvars = 7
# seq_len = 384
# d_model = 64
# pred_len = 196

# config= {'seq_len': seq_len,
#          'pred_len': 196,
#         'd_model': d_model,
#          }

# model = Model(config=None)

# sample_x = torch.tensor(ds.__getitem__(9)[0], dtype = torch.float32).unsqueeze(0)

# output = model(sample_x) # (bsz, nvars, seq_len, d_model)

# output.shape


# import torch

# sample = torch.randn(1, 384, 7)
# stats = torch.randn(1, 1, 7)

# mod = Projector(enc_in = 7, seq_len = 384, hidden_dims=[64, 64], hidden_layers = 2, output_dim=1, kernel_size=3)

# mod(sample, stats)

# series_conv = nn.Conv1d(in_channels=384, 
#                         out_channels=1, 
#                         kernel_size=3, padding=1, 
#                         padding_mode='circular', bias=False)


# series_conv(sample).shape



# x = series_conv(sample)
# x = torch.cat([x, stats], dim=1)
# x = x.view(bsz, -1)

# x.shape

# mod
