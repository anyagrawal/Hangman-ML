# trained_model.py
import torch
import torch.nn as nn

class HangmanPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.alphabet_size = 26
        self.blank_token = 1  # for '-', representing unknowns
        self.input_len = 20
        self.embedding_dim = 32
        self.lstm_hidden = 64
        self.word_repr_dim = 128  # output from LSTM + guessed vector

        self.embedding = nn.Embedding(self.alphabet_size + self.blank_token, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(self.lstm_hidden * 2 + self.alphabet_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, self.alphabet_size)  # output logits for 26 letters

    def forward(self, x):
        # x: (batch_size, input_dim) where input_dim = 20*27 + 26 = 566
        word_flat = x[:, :540]  # (batch_size, 540)
        guessed_vec = x[:, 540:]  # (batch_size, 26)

        batch_size = x.size(0)
        word_reshaped = word_flat.view(batch_size, self.input_len, 27)
        indices = torch.argmax(word_reshaped, dim=2)  # (batch_size, 20)

        embeds = self.embedding(indices)  # (batch_size, 20, embedding_dim)
        lstm_out, _ = self.lstm(embeds)  # (batch_size, 20, hidden*2)
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden*2)

        combined = torch.cat([last_hidden, guessed_vec], dim=1)  # (batch_size, hidden*2 + 26)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.fc2(x)  # (batch_size, 26)
        return x
