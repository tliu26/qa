import torch
import torch.nn as nn
import torch.nn.functional as F
from util import masked_softmax

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoding(nn.Module):
    """Encoding layer used by R-net, with character-level embedding.

    First look up word and character vectors. Then apply a bi-directional RNN
    to the character embeddings (now word_len is the seq_len in the usual RNN),
    use the last hidden state as the "character embedding" of the whole word.
    Finally use the concatenation of word and "character embeddings" as the
    representation of a word and apply an RNN to the sequence of words

    Parameters:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained char vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.

    Inputs:
        w_idxs (tensor) of shape (batch_size, seq_len).
        c_idxs (tensor) of shape (batch_size, seq_len, w_len).

    Outputs:
        u (tensor) of shape (batch_size, seq_len, 2 * hidden_size).
        s_lengths (tensor) of shape (batch_size).
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Encoding, self).__init__()
        self.drop_prob = drop_prob
        self.w_embed = nn.Embedding.from_pretrained(word_vectors)
        self.c_embed = nn.Embedding.from_pretrained(char_vectors)
        # RNN for refining character-level embedding within each word
        self.c_rnn = BiRNN(char_vectors.size(1), hidden_size, 1, drop_prob)
        # RNN for refining word and char embeddings of each sequence
        self.s_rnn = BiRNN(word_vectors.size(1) + 2*hidden_size, hidden_size,
                           3, drop_prob)
        self.out_size = hidden_size * 2

    def forward(self, w_idxs, c_idxs):
        # Get information about sizes of the text
        batch_size, seq_len, w_len = c_idxs.size()
        w_mask = torch.zeros_like(w_idxs) != w_idxs  # (batch_size, seq_len)
        s_lengths = w_mask.sum(-1)

        # Only work with elements of c_idxs for which the word is not empty,
        # that is, w_mask == 1. This means we take out c_idxs[i, j] for which
        # c_idxs[i, j] == 0 for all w_len elements in it.
        c_idxs_nonzeros = c_idxs[w_mask]  # (..., w_len)
        # (..., w_len, c_emb_size)
        c_emb_nonzeros = self.c_embed(c_idxs_nonzeros)
        w_lengths = (torch.zeros_like(c_idxs_nonzeros) !=
                     c_idxs_nonzeros).sum(-1)

        # get embeddings
        w_emb = self.w_embed(w_idxs)  # (batch_size, seq_len, w_emb_size)
        # (2, ..., hidden_size)
        _, h_n_nonzeros = self.c_rnn(c_emb_nonzeros, w_lengths)
        h_n = torch.zeros((2, batch_size, seq_len, h_n_nonzeros.size(-1)))
        h_n = h_n.to(self.device)
        h_n[:, w_mask] = h_n_nonzeros
        # Eq. (1) in [R-net]
        w_c_emb = torch.cat((w_emb, h_n[0], h_n[1]), dim=2)
        # u.shape = (batch_size, seq_len, 2 * hidden_size)
        u, _ = self.s_rnn(w_c_emb, s_lengths)
        u = u.permute([1, 0, 2])
        del h_n
        return u, s_lengths, w_mask

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.c_embed.weight.device


# Using passage and question to obtain question-aware passage representation
# Co-attention
class PQMatcher(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(PQMatcher, self).__init__()
        self.hidden_size = hidden_size * 2
        self.in_size = in_size
        self.gru = nn.GRUCell(input_size=in_size*2, hidden_size=self.hidden_size)
        self.Wp = nn.Linear(self.in_size*2, self.hidden_size, bias=False)
        self.Wq = nn.Linear(self.in_size*2, self.hidden_size, bias=False)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wg = nn.Linear(self.in_size*4, self.in_size*4, bias=False)
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, up, uq):
        (lp, batch_size, _) = up.size()
        (lq, batch_size, _) = uq.size()
        mixerp, mixerq = torch.arange(lp).long().to(self.device), torch.arange(lq).long().to(self.device)
        Up = torch.cat([up, up[mixerp]], dim=2)
        Uq = torch.cat([uq, uq[mixerq]], dim=2)
        vs = torch.zeros(lp, batch_size, self.out_size).to(self.device)
        v = torch.randn(batch_size, self.hidden_size).to(self.device)
        V = torch.randn(batch_size, self.hidden_size, 1).to(self.device)

        Uq_ = Uq.permute([1, 0, 2])
        for i in range(lp):
            Wup = self.Wp(Up[i])
            Wuq = self.Wq(Uq)
            Wvv = self.Wv(v)
            # (batch, seq_len, hidden)
            x = torch.tanh(Wup + Wuq + Wvv).permute([1, 0, 2])
            # (batch, seq_len, 1) -> (batch, seq_len)
            s = torch.bmm(x, V)
            s = torch.squeeze(s, 2)
            # (batch, 1, seq_len) @ (batch, seq_len, hidden_size)
            a = F.softmax(s, 1).unsqueeze(1)
            c = torch.bmm(a, Uq_).squeeze()
            r = torch.cat([Up[i], c], dim=1)
            g = torch.sigmoid(self.Wg(r))
            r_ = torch.mul(g, r)
            c_ = r_[:, self.in_size*2:]
            v = self.gru(c_, v)
            vs[i] = v
            del Wup, Wuq, Wvv, x, a, s, c, g, r, r_, c_
        del up, uq, Up, Uq, Uq_
        vs = self.dropout(vs)
        return vs

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.Wp.weight.device


# Input is question-aware passage representation
# Output is self-attention question-aware passage representation
class SelfMatcher(nn.Module):
    def __init__(self, in_size, dropout):
        super(SelfMatcher, self).__init__()
        self.hidden_size = in_size
        self.in_size = in_size
        self.gru = nn.GRUCell(input_size=in_size, hidden_size=self.hidden_size)
        self.Wp = nn.Linear(self.in_size, self.hidden_size, bias=False)
        self.Wp_ = nn.Linear(self.in_size, self.hidden_size, bias=False)
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v):
        (l, batch_size, _) = v.size()
        h = torch.randn(batch_size, self.hidden_size).to(self.device)
        V = torch.randn(batch_size, self.hidden_size, 1).to(self.device)
        hs = torch.zeros(l, batch_size, self.out_size).to(self.device)

        for i in range(l):
            Wpv = self.Wp(v[i])
            Wpv_ = self.Wp_(v)
            x = torch.tanh(Wpv + Wpv_)
            x = x.permute([1, 0, 2])
            s = torch.bmm(x, V)
            s = torch.squeeze(s, 2)
            a = F.softmax(s, 1).unsqueeze(1)
            c = torch.bmm(a, v.permute([1, 0, 2])).squeeze()
            h = self.gru(c, h)
            hs[i] = h
            # logger.gpu_mem_log("SelfMatcher {:002d}".format(i), ['x', 'Wpv', 'Wpv_', 's', 'c', 'hs'], [x.data, Wpv.data, Wpv_.data, s.data, c.data, hs.data])
            del Wpv, Wpv_, x, s, a, c
        hs = self.dropout(hs)
        del h, v
        return hs

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.Wp.weight.device


# Input is question representation and self-attention question-aware passage representation
# Output are start and end pointer distribution
class Pointer(nn.Module):
    def __init__(self, in_size1, in_size2):
        super(Pointer, self).__init__()
        self.hidden_size = in_size2
        self.in_size1 = in_size1
        self.in_size2 = in_size2
        self.gru = nn.GRUCell(input_size=in_size1, hidden_size=self.hidden_size)
        # Wu uses bias. See formula (11). Maybe Vr is just a bias.
        self.Wu = nn.Linear(self.in_size2, self.hidden_size, bias=True)
        self.Wh = nn.Linear(self.in_size1, self.hidden_size, bias=False)
        self.Wha = nn.Linear(self.in_size2, self.hidden_size, bias=False)
        self.out_size = 1

    def forward(self, h, u, c_mask):
        (lp, batch_size, _) = h.size()
        (lq, _, _) = u.size()
        v = torch.randn(batch_size, self.hidden_size, 1).to(self.device)
        u_ = u.permute([1,0,2])
        h_ = h.permute([1,0,2])
        x = torch.tanh(self.Wu(u)).permute([1, 0, 2])
        s = torch.bmm(x, v)
        s = torch.squeeze(s, 2)
        a = F.softmax(s, 1).unsqueeze(1)
        r = torch.bmm(a, u_).squeeze()
        x = torch.tanh(self.Wh(h)+self.Wha(r)).permute([1, 0, 2])
        s = torch.bmm(x, v)
        s = torch.squeeze(s)
        # p1 = F.log_softmax(s, 1)
        p1 = masked_softmax(s, c_mask, dim=1, log_softmax=True)
        c = torch.bmm(p1.unsqueeze(1), h_).squeeze()
        r = self.gru(c, r)
        x = torch.tanh(self.Wh(h) + self.Wha(r)).permute([1, 0, 2])
        s = torch.bmm(x, v)
        s = torch.squeeze(s)
        p2 = F.log_softmax(s, 1)
        p2 = masked_softmax(s, c_mask, dim=1, log_softmax=True)
        return (p1, p2)

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.Wu.weight.device


class BiRNN(nn.Module):
    """General purpose bi-directional gated recurrent unit (GRU) RNN.

    Parameters:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.

    Inputs:
        x (tensor) of shape (batch_size, seq_len, input_size).
        lengths (tensor, int) of shape (batch_size).

    Returns:
        output (tensor) of shape (batch_size, seq_len, 2 * hidden_size).
        h_n (tensor) fo shape (num_layers * 2, batch_size, hidden_size).
    """
    def __init__(self, input_size, hidden_size, num_layers, drop_prob):
        super(BiRNN, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        self.rnn.flatten_parameters()
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        # output.shape = (batch_size, seq_len, 2 * hidden_size)
        # h_n.shape = (num_layers * 2, batch_size, hidden_size)
        output, h_n = self.rnn(x)

        # Unpack and reverse sort
        output, _ = pad_packed_sequence(output, batch_first=True,
                                        total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        output = output[unsort_idx]
        h_n = h_n[:, unsort_idx]

        # Apply dropout (RNN applies dropout after all but the last layer)
        output = F.dropout(output, self.drop_prob, self.training)
        h_n = F.dropout(h_n, self.drop_prob, self.training)

        return output, h_n


# Using passage and question to obtain question-aware passage representation
# Co-attention
class PQMatcher1(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(PQMatcher1, self).__init__()
        self.hidden_size = hidden_size * 2
        self.in_size = in_size
        self.krnl = nn.Linear(self.in_size * 2, self.in_size * 2, bias=False)
        self.gru = nn.GRUCell(input_size=in_size*2, hidden_size=self.hidden_size)
        self.Wg = nn.Linear(self.in_size*4, self.in_size*4, bias=False)
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, up, uq):
        (lp, batch_size, _) = up.size()
        (lq, batch_size, _) = uq.size()
        mixerp, mixerq = torch.arange(lp).long().to(self.device), torch.arange(lq).long().to(self.device)
        Up = torch.cat([up, up[mixerp]], dim=2)
        Uq = torch.cat([uq, uq[mixerq]], dim=2)
        vs = torch.zeros(lp, batch_size, self.out_size).to(self.device)
        v = torch.randn(batch_size, self.hidden_size).to(self.device)

        Uq_ = Uq.permute([1, 0, 2])
        for i in range(lp):
            # (batch, seq_len, hidden_size) @ (batch, hidden_size, 1) ->
            # (batch, seq_len, 1) -> (batch, seq_len)

            # s = torch.bmm(Uq_, Up[i].unsqueeze(-1)).squeeze(-1)  # dot product
            # s = torch.bmm(Uq_, self.krnl(Up[i]).unsqueeze(-1)).squeeze(-1)  # linear kernel
            s = -(torch.norm(Uq_ - Up[i].unsqueeze(1), dim=-1) ** 2)  # L2-norm
            a = F.softmax(s, 1).unsqueeze(1)

            # (batch, 1, seq_len) @ (batch, seq_len, hidden_size)
            # a is the attention weights
            c = torch.bmm(a, Uq_).squeeze()
            r = torch.cat([Up[i], c], dim=1)
            g = torch.sigmoid(self.Wg(r))
            r_ = torch.mul(g, r)
            c_ = r_[:, self.in_size*2:]
            v = self.gru(c_, v)
            vs[i] = v
            del a, s, c, g, r, r_, c_
        del up, uq, Up, Uq, Uq_
        vs = self.dropout(vs)
        return vs

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.Wg.weight.device


# Input is question-aware passage representation
# Output is self-attention question-aware passage representation
class SelfMatcher1(nn.Module):
    def __init__(self, in_size, dropout):
        super(SelfMatcher1, self).__init__()
        self.hidden_size = in_size
        self.in_size = in_size
        self.krnl = nn.Linear(self.in_size, self.in_size, bias=False)
        self.gru = nn.GRUCell(input_size=in_size, hidden_size=self.hidden_size)
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v):
        (l, batch_size, _) = v.size()
        h = torch.randn(batch_size, self.hidden_size).to(self.device)
        hs = torch.zeros(l, batch_size, self.out_size).to(self.device)

        for i in range(l):
            # (batch, seq_len, hidden_size) @ (batch, hidden_size, 1) ->
            # (batch, seq_len, 1) -> (batch, seq_len)

            # s = torch.bmm(v.permute([1, 0, 2]), v[i].unsqueeze(-1)).squeeze(-1)  # dot product
            # s = torch.bmm(v.permute([1, 0, 2]), self.krnl(v[i]).unsqueeze(-1)).squeeze(-1)  # linear kernel
            s = -(torch.norm(v.permute([1, 0, 2]) - v[i].unsqueeze(1), dim=-1) ** 2)  # L2-norm
            a = F.softmax(s, 1).unsqueeze(1)
            c = torch.bmm(a, v.permute([1, 0, 2])).squeeze()
            h = self.gru(c, h)
            hs[i] = h
            # logger.gpu_mem_log("SelfMatcher {:002d}".format(i), ['x', 'Wpv', 'Wpv_', 's', 'c', 'hs'], [x.data, Wpv.data, Wpv_.data, s.data, c.data, hs.data])
            del s, a, c
        hs = self.dropout(hs)
        del h, v
        return hs

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.gru.weight_ih.device
