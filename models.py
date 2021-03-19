"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import r_net_layers
import rnl
import torch
import torch.nn as nn
import time


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAFCh(nn.Module):
    """Baseline BiDAF model *with* character-level embedding for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, char_channel_size,
                 char_channel_width, hidden_size, drop_prob=0.):
        super(BiDAFCh, self).__init__()

        self.emb = layers.WordCharEmbedding(word_vectors=word_vectors,
                                            char_vectors=char_vectors,
                                            char_channel_size=char_channel_size,
                                            char_channel_width=char_channel_width,
                                            hidden_size=hidden_size,
                                            drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs) # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs) # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class RNet(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(RNet, self).__init__()
        self.enc = r_net_layers.Encoding(word_vectors, char_vectors,
                                         hidden_size, drop_prob)
        # self.enc = r_net_layers.Encoding1(word_vectors, char_vectors,
        #                                  hidden_size, drop_prob)
        self.gan = r_net_layers.GatedAttnRNN(hidden_size, drop_prob)
        self.san = r_net_layers.SelfAttnRNN(hidden_size, drop_prob)
        self.out = r_net_layers.OutputLayer(hidden_size, drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        t0 = time.time()
        c_emb, c_len = self.enc(cw_idxs, cc_idxs)
        q_emb, q_len = self.enc(qw_idxs, qc_idxs)
        t1 = time.time()
        vp = self.gan(q_emb, c_emb, q_len, c_len)
        t2 = time.time()
        hp = self.san(vp, c_len)
        t3 = time.time()
        out = self.out(q_emb, hp, q_len, c_len)
        t4 = time.time()
        # print(f"Encoding: {t1 - t0} s")
        # print(f"GAN: {t2 - t1} s")
        # print(f"SAN: {t3 - t2} s")
        # print(f"Out: {t4 - t3} s")
        return out


class RNet2(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(RNet2, self).__init__()
        self.enc = r_net_layers.Encoding2(word_vectors, char_vectors,
                                          hidden_size, drop_prob)
        self.gan = r_net_layers.GatedAttnRNN2(hidden_size, drop_prob)
        self.san = r_net_layers.SelfAttnRNN2(hidden_size, drop_prob)
        self.out = r_net_layers.OutputLayer2(hidden_size, drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        t0 = time.time()
        c_emb, c_len, c_mask = self.enc(cw_idxs, cc_idxs)
        q_emb, q_len, q_mask = self.enc(qw_idxs, qc_idxs)
        t1 = time.time()
        vp = self.gan(q_emb, c_emb, q_mask, c_len)
        torch.cuda.empty_cache()
        t2 = time.time()
        hp = self.san(vp, c_mask, c_len)
        torch.cuda.empty_cache()
        t3 = time.time()
        out = self.out(q_emb, hp, q_mask, c_mask)
        t4 = time.time()
        # print(f"Encoding: {t1 - t0} s")
        # print(f"GAN: {t2 - t1} s")
        # print(f"SAN: {t3 - t2} s")
        # print(f"Out: {t4 - t3} s")
        return out


class RNet1(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(RNet1, self).__init__()
        char_channel_size = 100
        char_channel_width = 5
        self.enc = rnl.Encoding(word_vectors, char_vectors,
                                hidden_size, drop_prob)
        self.pqmatcher = rnl.PQMatcher1(self.enc.out_size, hidden_size, drop_prob)
        self.selfmatcher = rnl.SelfMatcher1(self.pqmatcher.out_size, drop_prob)
        self.pointer = rnl.Pointer(self.selfmatcher.out_size,
                                   self.enc.out_size)

        # self.enc = layers.WordCharEmbedding(word_vectors=word_vectors,
        #                                     char_vectors=char_vectors,
        #                                     char_channel_size=char_channel_size,
        #                                     char_channel_width=char_channel_width,
        #                                     hidden_size=hidden_size,
        #                                     drop_prob=drop_prob)
        # self.pqmatcher = rnl.PQMatcher(self.enc.out_size, hidden_size, drop_prob)
        # self.selfmatcher = rnl.SelfMatcher(self.pqmatcher.out_size, drop_prob)
        # self.pointer = rnl.Pointer(self.selfmatcher.out_size,
        #                            self.enc.out_size)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        t0 = time.time()
        c_emb, c_len, c_mask = self.enc(cw_idxs, cc_idxs)
        q_emb, q_len, q_mask = self.enc(qw_idxs, qc_idxs)
        # c_emb = self.enc(cw_idxs, cc_idxs).permute([1, 0, 2])
        # q_emb = self.enc(qw_idxs, qc_idxs).permute([1, 0, 2])
        t1 = time.time()
        v = self.pqmatcher(c_emb, q_emb)
        t2 = time.time()
        torch.cuda.empty_cache()
        h = self.selfmatcher(v)
        t3 = time.time()
        p1, p2 = self.pointer(h, q_emb, c_mask)
        t4 = time.time()
        # print(f"Encoding: {t1 - t0} s")
        # print(f"pqmatcher: {t2 - t1} s")
        # print(f"selfmatcher: {t3 - t2} s")
        # print(f"pointer: {t4 - t3} s")
        return p1, p2
