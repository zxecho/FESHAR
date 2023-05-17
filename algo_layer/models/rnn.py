import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 16


class LSTMNet(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2,
                 padding_idx=0, vocab_size=98635, num_classes=10):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        dims = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        text, text_lengths = x

        embedded = self.embedding(text)

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)

        return out
