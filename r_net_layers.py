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
        return u, s_lengths

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.c_embed.weight.device


class GatedAttnRNN(nn.Module):
    """Gated attention-based recurrent networks.

    Parameters:
        hidden_size (int): Size of the RNN hidden state.
        drop_prob (float): Probability of zero-ing out activations.

    Inputs:
        q_enc (tensor) of shape (batch_size, q_seq_len, 2 * hidden_size).
        p_enc (tensor) of shape (batch_size, p_seq_len, 2 * hidden_size).
        p_s_lengths (tensor) of shape (batch_size).

    Outputs:
        vp (tensor) of shape (batch_size, seq_len, 2 * hidden_size).
    """
    def __init__(self, hidden_size, drop_prob):
        super(GatedAttnRNN, self).__init__()
        self.linear_uQ = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.linear_uP = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.linear_vP = nn.Linear(hidden_size,     hidden_size, bias=False)
        self.linear_vT = nn.Linear(hidden_size,               1, bias=False)
        self.GRUcell = nn.GRUCell(hidden_size * 4, hidden_size, bias=True)
        self.g = nn.Linear(hidden_size * 4, hidden_size * 4, bias=False)
        self.drop_prob = drop_prob

    def attn(self, q_enc, p_enc_t, q_s_lengths, vp_t=0):
        """Calculate the attention-pooling vector, see Eq. (4) in [R-net].

        Inputs:
            q_enc (tensor) of shape (batch_size, q_seq_len, 2 * hidden_size).
            p_enc_t (tensor) of shape (batch_size, 2 * hidden_size).
            q_s_lengths (tensor) of shape (batch_size)
            vp_t (tensor) of shape (batch_size, hidden_size).

        Outputs:
            ct (tensor) of shape (batch_size, 2 * hidden_size).

        TODO: Can/Should we optimize the padded q_enc, i. e. don't calculate
        the linear and softmax for q_enc[:, s] where it's all zeros?
        """
        # (batch_size, q_seq_len, 1)
        if type(vp_t) == torch.Tensor:
            st = self.linear_vT(torch.tanh(self.linear_uQ(q_enc)
                                           + self.linear_uP(p_enc_t)[:, None]
                                           + self.linear_vP(vp_t)[:, None]))
        elif vp_t == 0:
            st = self.linear_vT(torch.tanh(self.linear_uQ(q_enc)
                                           + self.linear_uP(p_enc_t)[:, None]))
        else:
            raise ValueError("Uh-oh")
        q_seq_len = q_enc.size(1)
        q_mask = [[1] * length.item() + [0] * (q_seq_len - length.item()) for
                  length in q_s_lengths]
        q_mask = torch.tensor(q_mask)[..., None]  # (batch_size, q_seq_len, 1)
        q_mask = q_mask.to(self.device)
        at = masked_softmax(st, q_mask, dim=1)  # (batch_size, q_seq_len, 1)
        # at = F.softmax(st, dim=1)  # (batch_size, q_seq_len, 1)
        ct = (at * q_enc).sum(1)  # (batch_size, 2 * hidden_size)
        ct = F.dropout(ct, self.drop_prob, self.training)
        return ct

    def forward(self, q_enc, p_enc, q_s_lengths, p_s_lengths):
        batch_size, seq_len, hidden_sizex2 = p_enc.size()
        hidden_size = hidden_sizex2 // 2
        # (batch_size, p_seq_len, 2 * hidden_size)
        rev_p_enc = reverse_enc(p_enc, p_s_lengths)
        # (2 * batch_size, p_seq_len, 2 * hidden_size)
        bidir_p_enc = torch.cat((p_enc, rev_p_enc), dim=0)
        # (2 * batch_size)
        bidir_p_s_lengths = torch.cat((p_s_lengths, p_s_lengths), dim=0)
        packed_bidir_p_enc, sorted_p_s_lengths, sort_idx =\
            sort_pack_seq(bidir_p_enc, bidir_p_s_lengths)
        ppe_data = packed_bidir_p_enc.data
        ppe_batch_sizes = packed_bidir_p_enc.batch_sizes
        vp = torch.zeros((2 * batch_size, seq_len, hidden_size)).\
            to(self.device)
        # Need to duplicate q_enc and then sort according to how batch elements
        # in p_enc are sorted, in order to match the input to the attention.
        q_enc2_sorted = torch.cat((q_enc, q_enc), dim=0)[sort_idx]
        q_s_lengths2 = torch.cat((q_s_lengths, q_s_lengths), dim=0)[sort_idx]
        # The longest passage length must be equal to the sequence length in
        # the packed sequence
        assert(p_s_lengths.max() == ppe_batch_sizes.size(0))
        s_idx = 0
        for t, cur_batch_size in enumerate(ppe_batch_sizes):
            e_idx = s_idx + cur_batch_size
            # (cur_batch_size, hidden_size * 2)
            up_t = ppe_data[s_idx:e_idx]
            # (cur_batch_size, hidden_size * 2)
            if t == 0:
                ct = self.attn(q_enc2_sorted[:cur_batch_size], up_t,
                               q_s_lengths2[:cur_batch_size])
            else:
                ct = self.attn(q_enc2_sorted[:cur_batch_size], up_t,
                               q_s_lengths2[:cur_batch_size],
                               vp[:cur_batch_size, t-1].clone())
            # (cur_batch_size, hidden_size * 4)
            upc_t = torch.cat((up_t, ct), dim=-1)
            # Eq. (6) in [R-net]
            gt = torch.sigmoid(self.g(upc_t))
            vp[:cur_batch_size, t] =\
                self.GRUcell(gt * upc_t, vp[:cur_batch_size, t-1].clone())
            s_idx = e_idx
        _, unsort_idx = sort_idx.sort(0)
        vp = vp[unsort_idx]
        vp = vp.view(2, batch_size, seq_len, hidden_size)
        # (batch_size, seq_len, 2 * hidden_size).
        vp = torch.cat((vp[0], vp[1]), dim=-1)
        vp = F.dropout(vp, self.drop_prob, self.training)
        return vp

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.linear_vT.weight.device


class SelfAttnRNN(nn.Module):
    """Gated self-attention recurrent networks.

    Parameters:
        hidden_size (int): Size of the RNN hidden state.
        drop_prob (float): Probability of zero-ing out activations.

    Inputs:
        vp (tensor) of shape (batch_size, p_seq_len, 2 * hidden_size).
        p_s_lengths (tensor) of shape (batch_size).

    Outputs:
        hp (tensor) of shape (batch_size, seq_len, 2 * hidden_size).
    """
    def __init__(self, hidden_size, drop_prob):
        super(SelfAttnRNN, self).__init__()
        self.linear_vP = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.linear_vPt = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.linear_vT = nn.Linear(hidden_size,               1, bias=False)
        self.GRUcell = nn.GRUCell(hidden_size * 4, hidden_size, bias=True)
        self.g = nn.Linear(hidden_size * 4, hidden_size * 4, bias=False)
        self.drop_prob = drop_prob

    def attn(self, vp, vp_t, p_s_lengths):
        """Calculate the self attention-pooling vector, see Eq. (8) in [R-net]

        Inputs:
            vp (tensor) of shape (batch_size, seq_len, 2 * hidden_size)
            vp_t (tensor) of shape (batch_size, 2 * hidden_size)
            p_s_lengths (tensor) of shape (batch_size)

        Outputs:
            ct (tensor) of shape (batch_size, 2 * hidden_size)

        TODO: Can/Should we optimize the padded vp, i. e. don't calculate
        the linear and softmax for vp[:, s] where it's all zeros?
        """
        # (batch_size, q_seq_len, 1)
        st = self.linear_vT(torch.tanh(self.linear_vP(vp)
                                       + self.linear_vPt(vp_t)[:, None]))
        seq_len = vp.size(1)
        p_mask = [[1] * length.item() + [0] * (seq_len - length.item()) for
                  length in p_s_lengths]
        p_mask = torch.tensor(p_mask)[..., None]  # (batch_size, q_seq_len, 1)
        # (batch_size, q_seq_len, 1)
        at = masked_softmax(st, p_mask.to(self.device), dim=1)
        ct = (at * vp).sum(1)  # (batch_size, 2 * hidden_size)
        ct = F.dropout(ct, self.drop_prob, self.training)
        return ct

    def forward(self, vp, p_s_lengths):
        batch_size, seq_len, hidden_sizex2 = vp.size()
        hidden_size = hidden_sizex2 // 2
        # (batch_size, p_seq_len, 2 * hidden_size)
        rev_vp = reverse_enc(vp, p_s_lengths)
        # (2 * batch_size, p_seq_len, 2 * hidden_size)
        bidir_vp = torch.cat((vp, rev_vp), dim=0)
        # (2 * batch_size)
        bidir_p_s_lengths = torch.cat((p_s_lengths, p_s_lengths), dim=0)
        packed_bidir_vp, sorted_p_s_lengths, sort_idx =\
            sort_pack_seq(bidir_vp, bidir_p_s_lengths)
        pvp_data = packed_bidir_vp.data
        pvp_batch_sizes = packed_bidir_vp.batch_sizes
        hp = torch.zeros((2 * batch_size, seq_len, hidden_size)).\
            to(self.device)
        # Need to duplicate vp and then sort according to how batch_elements in
        # vp are sorted, in order to match the key and query in attention.
        vp2_sorted = torch.cat((vp, vp), dim=0)[sort_idx]
        p_s_lengths2 = bidir_p_s_lengths[sort_idx]
        # The longest passage length must be equal to the sequence length in
        # the packed sequence
        assert(p_s_lengths.max() == pvp_batch_sizes.size(0))
        s_idx = 0
        for t, cur_batch_size in enumerate(pvp_batch_sizes):
            e_idx = s_idx + cur_batch_size
            # (cur_batch_size, hidden_size * 2)
            vp_t = pvp_data[s_idx:e_idx]
            # (cur_batch_size, hidden_size * 2)
            # print(vp2_sorted.device, vp_t.device, p_s_lengths2.device)
            ct = self.attn(vp2_sorted[:cur_batch_size].to(self.device),
                           vp_t.to(self.device), p_s_lengths2[:cur_batch_size])
            # (cur_batch_size, hidden_size * 4)
            vpc_t = torch.cat((vp_t.to(self.device), ct), dim=-1)
            # An additional gate like Eq. (6) in [R-net]
            gt = torch.sigmoid(self.g(vpc_t))
            hp[:cur_batch_size, t] =\
                self.GRUcell(gt * vpc_t, hp[:cur_batch_size, t-1].clone())
            s_idx = e_idx
        _, unsort_idx = sort_idx.sort(0)
        hp = hp[unsort_idx]
        hp = hp.view(2, batch_size, seq_len, hidden_size)
        # (batch_size, seq_len, 2 * hidden_size).
        hp = torch.cat((hp[0], hp[1]), dim=-1)
        hp = F.dropout(hp, self.drop_prob, self.training)
        return hp

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.linear_vT.weight.device


def reverse_enc(enc, s_lengths):
    """Give the encodings in reverse order along the seq_len dimension,
    reversing only up to the actual sequence length (give by s_lengths) for
    each batch element.

    Inputs:
        enc (tensor) of shape (batch_size, q_seq_len, ...).
        s_lengths (tensor) of shape (batch_size).

    Outputs:
        rev_enc (tensor) of shape (batch_size, q_seq_len, ...).
    """
    seq_len = enc.size(1)
    rev_enc = []
    for enc_b, s_len in zip(enc, s_lengths):
        rev_idx = list(reversed(range(s_len))) + list(range(s_len, seq_len))
        rev_enc.append(enc_b[rev_idx][None])
    rev_enc = torch.cat(rev_enc, 0)
    return rev_enc


def sort_pack_seq(enc, lengths):
    """Sort the sequence by lengths and pack the sequence.

    Inputs:
        enc (tensor) of shape (batch_size, seq_len, ...).
        lengths (tensor) of shape (batch_size).

    Outputs:
        enc (PackedSequence) of shape (batch_size, seq_len, ...).
        lengths (tensor) of shape (batch_size).
        sort_idx (tensor) of shape (batch_size).
    """
    lengths, sort_idx = lengths.sort(0, descending=True)
    enc = enc[sort_idx]
    enc = pack_padded_sequence(enc, lengths, batch_first=True)
    return enc, lengths, sort_idx


class OutputLayer(nn.Module):
    """Calculate the start and end position probability distribution over the
    context using pointer networks. See Sec. (3.4) of [R-net]

    Parameters:
        hidden_size (int): Size of the RNN hidden state.
        drop_prob (float): Probability of zero-ing out activations.

    Inputs:
        uQ (tensor) of shape (batch_size, q_seq_len, hidden_size * 2)
        hp (tensor) of shape (batch_size, p_seq_len, hidden_size * 2)
        q_s_lengths (tensor) of shape (batch_size)
        p_s_lengths (tensor) of shape (batch_size)

    Outputs:
        log_p1 (tensor) of shape (batch_size, p_seq_len)
        log_p2 (tensor) of shape (batch_size, p_seq_len)
    """
    def __init__(self, hidden_size, drop_prob):
        super(OutputLayer, self).__init__()
        self.drop_prob = drop_prob
        self.linear_hP = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.linear_ha = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.linear_vT2 = nn.Linear(hidden_size, 1, bias=False)
        self.GRUcell = nn.GRUCell(hidden_size * 2, hidden_size * 2, bias=True)
        self.linear_uQ = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.linear_vQ = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.linear_vT1 = nn.Linear(hidden_size, 1, bias=False)

    def attn1(self, uQ, q_s_lengths):
        """Calculate the attention-pooling vector of the question. See Eq. (11)
        in [R-net]

        Inputs:
            uQ (tensor) of shape (batch_size, seq_len, 2 * hidden_size)
            q_s_lengths (tensor) of shape (batch_size)

        Outputs:
            rQ (tensor) of shape (batch_size, 2 * hidden_size)
        """
        # (batch_size, seq_len, 1)
        s = self.linear_vT1(torch.tanh(self.linear_uQ(uQ)))
        seq_len = uQ.size(1)
        q_mask = [[1] * length.item() + [0] * (seq_len - length.item()) for
                  length in q_s_lengths]
        # (batch_size, q_seq_len, 1)
        q_mask = torch.tensor(q_mask)[..., None].to(self.device)
        a = masked_softmax(s, q_mask, dim=1)
        rQ = (a * uQ).sum(1)  # (batch_size, 2 * hidden_size)
        rQ = F.dropout(rQ, self.drop_prob, self.training)
        return rQ

    def attn2(self, hp, ha_tm1, p_s_lengths):
        """Calculate attention weights used as pointer selector. See Eq. (9) in
        [R-net]

        Inputs:
            hp (tensor) of shape (batch_size, seq_len, hidden_sizde * 2)
            ha_tm1 (tensor) of shape (batch_size, 2 * hidden_size)
            p_s_lengths (tensor) of shape (batch_size)

        Outputs:
            at (tensor) of shape (batch_size, p_seq_len, 1)
        """
        st = self.linear_vT2(torch.tanh(self.linear_hP(hp)
                                        + self.linear_ha(ha_tm1)[:, None]))
        seq_len = hp.size(1)
        p_mask = [[1] * length.item() + [0] * (seq_len - length.item()) for
                  length in p_s_lengths]
        # (batch_size, p_seq_len, 1)
        p_mask = torch.tensor(p_mask)[..., None].to(self.device)
        at = masked_softmax(st, p_mask, dim=1, log_softmax=True)
        return at

    def forward(self, uQ, hp, q_s_lengths, p_s_lengths):
        rQ = self.attn1(uQ, q_s_lengths)  # (batch_size, 2 * hidden_size)
        log_p1 = self.attn2(hp, rQ, p_s_lengths)  # (batch_size, p_seq_len, 1)
        # Eq. (10) in [R-net]
        ct = (torch.exp(log_p1) * hp).sum(1)  # (batch_size, 2 * hidden_size)
        log_p1 = log_p1.squeeze(-1)  # (batch_size, p_seq_len)
        ha = self.GRUcell(ct, rQ)  # (batch_size, hidden_size * 2)
        # (batch_size, p_seq_len, 1)
        log_p2 = self.attn2(hp, ha, p_s_lengths).squeeze(-1)
        return log_p1, log_p2

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.linear_hP.weight.device


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


"""
References:
    [R-net]: https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
"""


class Encoding1(nn.Module):
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
        super(Encoding1, self).__init__()
        self.drop_prob = drop_prob
        self.w_embed = nn.Embedding.from_pretrained(word_vectors)
        self.c_embed = nn.Embedding.from_pretrained(char_vectors)
        # RNN for refining character-level embedding within each word
        self.c_rnn = BiRNN(char_vectors.size(1), hidden_size, 1, drop_prob)
        self.c_rnn = nn.GRU(char_vectors.size(1), hidden_size,
                            batch_first=True, bidirectional=True)
        # RNN for refining word and char embeddings of each sequence
        self.s_rnn = BiRNN(word_vectors.size(1) + 2*hidden_size, hidden_size,
                           3, drop_prob)

    def forward(self, w_idxs, c_idxs):
        # Get information about sizes of the text
        batch_size, seq_len, w_len = c_idxs.size()
        w_mask = torch.zeros_like(w_idxs) != w_idxs  # (batch_size, seq_len)
        s_lengths = w_mask.sum(-1)

        # get embeddings
        w_emb = self.w_embed(w_idxs)  # (batch_size, seq_len, w_emb_size)
        c_emb = self.c_embed(c_idxs).view(batch_size * seq_len, w_len, -1)
        _, h_n = self.c_rnn(c_emb)
        h_n = h_n.view(2, batch_size, seq_len, -1)
        # Eq. (1) in [R-net]
        w_c_emb = torch.cat((w_emb, h_n[0], h_n[1]), dim=2)
        # u.shape = (batch_size, seq_len, 2 * hidden_size)
        u, _ = self.s_rnn(w_c_emb, s_lengths)
        return u, s_lengths

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.c_embed.weight.device
