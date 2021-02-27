import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoding(nn.Module):
    """Encoding layer used by R-net, with character-level embedding.

    First look up word and character vectors. Then apply a bi-directional RNN to
    the character embeddings (now word_len is the seq_len in the usual RNN), use
    the last hidden state as the "character embedding" of the whole word.
    Finally use the concatenation of word and "character embeddings" as the
    representation of a word and apply an RNN to the sequence of words
    
    Parameters:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained char vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Encoding, self).__init__()
        self.drop_prob = drop_prob
        self.w_embed = nn.Embedding.from_pretrained(word_vectors)
        self.c_embed = nn.Embedding.from_pretrained(char_vectors)
        # RNN for refining character-level embedding within each word
        self.c_rnn = nn.GRU(char_vectors.size(1), hidden_size, num_layers=1,
                          batch_first=True, bidirectional=True)
        # RNN for refining word and char embeddings of each sequence
        self.s_rnn = BiRNN(word_vectors.size(1) + 2*hidden_size, hidden_size, 3,
                           drop_prob)
    
    def forward(self, w_idxs, c_idxs, s_lengths):
        batch_size, seq_len, w_len = c_idxs.size()
        w_emb = self.w_embed(w_idxs) # (batch_size, seq_len, w_emb_size)
        c_emb = self.c_embed(c_idxs) # (batch_size, seq_len, w_len, c_emb_size)
        c_emb = c_emb.view(batch_size * seq_len, w_len, -1)
        # h_n.shape = (2, batch_size * seq_len, hidden_size)
        _, h_n = self.c_rnn(c_emb)
        h_n = h_n.view(2, batch_size, seq_len, -1)
        # Eq. (1) in [R-net]
        w_c_emb = torch.cat((w_emb, h_n[0], h_n[1]), dim=2)
        # u.shape = (batch_size, seq_len, 2 * hidden_size)
        u, _ = self.s_rnn(w_c_emb, s_lengths)
        return u
        

class BiRNN(nn.Module):
    """General purpose bi-directional gated recurrent unit (GRU) RNN

    Parameters:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    
    Inputs:
        x (tensor) of shape (batch_size, seq_len, input_size)
        lengths (tensor, int) of shape (batch_size)
    
    Returns:
        output (tensor) of shape (batch_size, seq_len, 2 * hidden_size)
        h_n (tensor) fo shape (num_layers * 2, batch_size, hidden_size)
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