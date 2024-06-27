import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def XNorm(x, gamma):
    norm_tensor = torch.norm(x, 4, -1, True)
    return x * gamma / norm_tensor

class UFOAttention(nn.Module):
    '''
    UFO-ViT: High Performance Linear Vision Transformer without Softmax
    https://arxiv.org/abs/2110.07641
    '''
    def __init__(self, embed_size, heads, dropout):

        super(UFOAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.fc_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_o = nn.Linear(heads * self.head_dim, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.randn((1, heads, 1, 1)))

    def forward(self, queries, keys, values):
        # b_s, nq = queries.shape[:2]
        # nk = keys.shape[1]
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        # q = self.fc_q(queries).view(b_s, nq, self.heads, self.head_dim).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        # k = self.fc_k(keys).view(b_s, nk, self.heads, self.head_dim).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        # v = self.fc_v(values).view(b_s, nk, self.heads, self.head_dim).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        v = self.fc_v(values).permute(0, 2, 1, 3)  # (N, query_len, heads, heads_dim)
        k = self.fc_k(keys).permute(0, 2, 3, 1)
        q = self.fc_q(queries).permute(0, 2, 1, 3)

        kv = torch.matmul(k, v)          # bs,h,c,c
        kv_norm = XNorm(kv, self.gamma)  # bs,h,c,c
        q_norm = XNorm(q, self.gamma)    # bs,h,n,c
        out = torch.matmul(q_norm, kv_norm).permute(0, 2, 1, 3).contiguous().view(N, query_len, self.heads * self.head_dim)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed Size needs to be Divisible by Heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        self.softmax = nn.Softmax()

    def forward(self, values, keys, query):

        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        """Split Embedding into self.head pieces"""
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)#(N, query_len, heads, heads_dim)
        keys = self.keys(keys)
        queries = self.queries(queries)

        atten = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])
        atten = self.softmax(atten)
        out = torch.einsum("nhql, nlhd->nqhd", [atten, values]).reshape(N, query_len, self.heads * self.head_dim)# attention shape: (N, heads, query_len, key_len)
        out = self.fc_out(out)

        return out

class UFOSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(UFOSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed Size needs to be Divisible by Heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):

        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        """Split Embedding into self.head pieces"""
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)#(N, query_len, heads, heads_dim)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])

        energy_norm = XNorm(energy, 0.01)  # bs,h,c,c
        values_norm = XNorm(values, 0.01)  # bs,h,n,c
        out = torch.einsum("nhql, nlhd->nqhd", [energy_norm, values_norm]).reshape(N, query_len, self.heads * self.head_dim)# attention shape: (N, heads, query_len, key_len)
        out = self.fc_out(out)

        return out

class SQLayer1d(nn.Module):
    def __init__(self, embed_size):
        super(SQLayer1d, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.fc(x)
        return x * y.expand_as(x)

class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
        # self.pool = nn.MaxPool1d(pool_size,stride=1,padding=pool_size//2)

    def forward(self, x):
        return self.pool(x) - x

class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, h, C]
    """
    def    __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)

        s = (x - u).pow(2).mean(1, keepdim=True)# axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(0) * x + self.bias.unsqueeze(0)
        return x


class MeSFormerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, forwordatt_type='sq',atten_type='pool_att'):
        super(MeSFormerBlock, self).__init__()
        self.atten_type = atten_type
        if atten_type == 'self_att':
            self.attention = SelfAttention(embed_size=embed_size, heads=heads)
            self.norm1 = nn.LayerNorm(embed_size)
        elif atten_type == 'ufo_att':
            self.attention = UFOAttention(embed_size=embed_size, heads=heads)
            self.norm1 = nn.LayerNorm(embed_size)
        elif atten_type == 'ufo_self_att':
            self.attention = UFOSelfAttention(embed_size=embed_size, heads=heads)
            self.norm1 = nn.LayerNorm(embed_size)
        elif atten_type == 'pool_att':
            self.token_mixer = Pooling()
            # self.drop_path = DropPath(dropout)
            # self.dropout = nn.Dropout(dropout)

            self.norm1 = LayerNormChannel(embed_size)

        self.forwordatt_type = forwordatt_type
        if self.forwordatt_type == 'sq':
            self.sqlayer = SQLayer1d(embed_size)
        elif self.forwordatt_type == 'original':
            self.norm2 = nn.LayerNorm(embed_size)
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_size, forward_expansion * embed_size),
                nn.ReLU(),
                nn.Linear(forward_expansion * embed_size, embed_size)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        if self.atten_type == 'pool_att':
            # print(self.norm1(query).size(),'111111111')
            x = query + self.dropout(self.token_mixer(self.norm1(query)))
        else:
            attention = self.attention(value, key, query)
            x = self.dropout(self.norm1(attention + query))

        if self.forwordatt_type =='sq':
            x = self.sqlayer(x)
        elif self.forwordatt_type =='original':
            forward = self.feed_forward(x)
            x = self.dropout(self.norm2(forward + x))
            x = self.dropout(self.norm2(forward + x))

        return x



class MeSFormer(nn.Module):
    def __init__(self, embed_size=64, heads=4, num_layers=2, forward_expansion=4, dropout=0.3,
                 forwordatt_type='sq',atten_type='pool_att',device="cuda"):
        super(MeSFormer, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.trans_layers = nn.ModuleList(
            [MeSFormerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    forwordatt_type = forwordatt_type,
                    atten_type=atten_type
                )for _ in range(num_layers)])

    def forward(self, gcn):
        b, c, n = gcn.size()
        # print(gcn.size(),'44444444444')

        for layer in self.trans_layers:
            # Query, Key and Value same in Encoder
            out = layer(gcn, gcn, gcn)
        out = out.reshape(b, c * n)
        return out