"""The BiLSTM sentiment classifier (PyTorch).

    token ids -> Embedding -> Dropout -> LSTM -> final hidden state(s) -> Dropout -> Linear -> 3 scores
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=3,
                 bidirectional=True, dropout=0.3, pad_idx=0):
        super().__init__()
        self.bidirectional = bidirectional

        # Embedding: a lookup table with one learnable vector (100 numbers) per word id.
        # padding_idx=0 keeps the <pad> vector at zeros and never updates it.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Dropout randomly zeroes 30% of the numbers during training, so the model cannot
        # rely on any single number; this fights overfitting. It is switched off by model.eval().
        self.dropout = nn.Dropout(dropout)

        # LSTM: reads the tweet one token at a time and keeps a memory (the hidden state).
        # batch_first=True means inputs are shaped [batch, time, features].
        # bidirectional=True runs a second LSTM from the last token back to the first.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True,
                            bidirectional=bidirectional)

        # The final vector has hidden_dim numbers, or 2 * hidden_dim if both directions are joined
        final_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(final_dim, num_classes)   # turns the vector into 3 class scores (logits)

    def forward(self, ids, lengths):
        """ids: [batch, 40] word ids. lengths: [batch] real number of tokens of each tweet."""
        embedded = self.dropout(self.embedding(ids))            # [batch, 40, 100]

        # pack_padded_sequence: tweets are padded with <pad> up to 40 tokens. If the LSTM read
        # those padding steps, its memory would be polluted by meaningless tokens (and the
        # backward direction would START on padding). Packing tells the LSTM the real length of
        # each tweet so it stops after the last real token and ignores the padding.
        # lengths must be on the CPU; enforce_sorted=False lets the batch stay in any order.
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # We only need h_n, the hidden state after the LAST real token, shaped
        # [num_directions, batch, hidden]. Index 0 = forward LSTM, index 1 = backward LSTM.
        _, (h_n, _) = self.lstm(packed)

        if self.bidirectional:
            # forward LSTM has read the whole tweet left-to-right; backward right-to-left.
            # Glue the two summaries together: [batch, 2 * hidden]
            final = torch.cat([h_n[0], h_n[1]], dim=1)
        else:
            final = h_n[0]                                      # [batch, hidden]

        return self.fc(self.dropout(final))                     # [batch, 3]
