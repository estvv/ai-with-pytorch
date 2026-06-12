import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, n_layers,
                             batch_first=True, dropout=0.2)
        self.fc    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)
        x, h = self.lstm(x, h)
        x = self.fc(x)
        return x, h
