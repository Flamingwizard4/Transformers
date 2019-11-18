#Tutorial with code can be found here: http://nlp.seas.harvard.edu/2018/04/03/attention.html#background

###TODO:
#add training code
#debugging
#test code
#upgrade to Transformer XL
###

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time, os, pdb
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context(context="talk")
#%matplotlib inline

def clones(module, N):
    #produces N identical layers
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    #masks out subsequent positions, ensuring predictions for position i can depend only on the known outputs at positions less than i
    attn_shape = (1, size, size)
    sub_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(sub_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    #computes 'Scaled Dot Product Attention'
    d_k = query.size(-1) #dimension of queries/keys
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #attention scores
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn #returns weighted values and softmaxed attention scores

    """
    Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions.

    In this work we employ h = 8 parallel attention layers, or heads.
    For each of these we use d_k = d_v = d_model/h = 64.

    Due to the reduced dimension of each head, the total computational cost
    is similar to that of single-head attention with full dimensionality (512).
    """

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Contructs model from hyperparameters:

    src/tgt vocab -
    N - # of layers stacked in encoder/decoder
    d_model - dimension of model/embedded vectors
    d_ff - dimension of inner-layer in feed_forward layer
    h - # of parallel attention layers aka "heads"
    dropout - rate of dropout
    """
    c = copy.deepcopy #deepcopy shortcut
    attn = MultiHeadAttn(h, d_model) #attention
    ff = PositionwiseFFLayer(d_model, d_ff, dropout) #feed forward layer
    pos = PositionalEncoding(d_model, dropout) #positional encoding
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(pos)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(pos)),
        Generator(d_model, tgt_vocab))

    #This was important/imported from their code
    #Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, enc, dec, src_embed, tgt_embed, gen):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        #take in and process masked src and tgt sequences
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask): #encode method framework, encoder/embedder passed in
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask): #decode method framework, decoder/embedder passed in
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    #defines standard linear + softmax generation step (the end I think)
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab) #torch.nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim = -1) #why the use of a different softmax here at the end??


class LayerNorm(nn.Module): #used in SublayerConnection & core Encoder/Decoder classes
    #constructs a layer normalization module
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        slef.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    #Core encoder is a stack of N layers
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        #passes the input (and mask) through each layer in turn
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    #Generic N layer decoder with masking
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class SublayerConnection(nn.Module): #used in Encoder/Decoder Layers
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        #applies residual connection to ny sublayer with the same size
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    #Encoder is made up of multi-head attention and feed-forward layers, with norms after
    def __init__(self, size, multihead, ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = multihead
        self.feed_forward = ff
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    #Decoder is made of self_attn, src_attn (multi-head attention over the output of the encoder stack), and ff
    def __init__(self, size, self_attn, src_attn, ff, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = ff
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadAttn(nn.Module):
    #used for decoding/src_attn, encoding/self_attn, and decoding/self_attn i.e. autoregression
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttn, self).__init__()
        assert d_model % heads == 0
        #We assume d_v always equals d_k
        self.d_k = d_model // heads
        self.h = heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            #same mask applied to all h heads
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)

        #do all linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(n_batches, -1, self.d_k).transpose(1, 2)
        for l, x in zip(self.linears, (query, key, value))]

        #apply attention on all projected vectors in the batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        #"concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)

        return self.linears[-1](x)

class PositionwiseFFLayer(nn.Module):
    """
    In addition to attention sub-layers, each of the layers in our encoder and decoder
    contains a fully connected feed-forward network, which is applied to each position
    separately and identically. This consists of two linear transformations with a
    ReLU activation in between.

    Implements FFN equation FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    While the linear transformations are the same across different positions, they use
    different parameters from layer to layer. Another way of describing this is as
    two convolutions with kernel size 1. The dimensionality of input and output is
    d_model = 512, and the inner-layer has dimensionality d_ff = 2048.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFFLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    """
    Similarly to other sequence transduction models, we use learned embeddings to
    convert the input tokens and output tokens to vectors of dimension d_model.
    We also use the usual learned linear transformation and softmax function to
    convert the decoder output to predicted next-token probabilities. In our model,
    we share the same weight matrix between the two embedding layers and the
    pre-softmax linear transformation. In the embedding layers, we multiply those weights by √d_model.
    """
    def __init__(self, d_model, vocab): #same variables as Generator
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model) #what does lut stand for??
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Since our model contains no recurrence and no convolution, in order for the model to
    make use of the order of the sequence, we must inject some information about the
    relative or absolute position of the tokens in the sequence. To this end, we add “positional encodings”
    to the input embeddings at the bottoms of the encoder and decoder stacks.

    The positional encodings have the same dimension d_model as the embeddings, so that the two can be summed.
    There are many choices of positional encodings, learned and fixed.

    In this work, we use sine and cosine functions of different frequencies:
    PE(pos,2i) = sin(pos/10000^2i/dmodel)
    PE(pos,2i+1) = cos(pos/10000^2i/dmodel)
    Where pos is the position and i is the dimension.

    That is, each dimension of the positional encoding corresponds to a sinusoid.
    The wavelengths form a geometric progression from 2π to 10000⋅2π. We chose this function because we
    hypothesized it would allow the model to easily learn to attend by relative positions,
    since for any fixed offset k, PE_pos+k can be represented as a linear function of PE_pos.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) #this is a LongTensor instead of a FloatTensor for some reason
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


tmp_model = make_model(10, 10, 2)
