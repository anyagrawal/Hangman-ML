# trained_model.py
import string
import torch.nn as nn
import numpy as np

class HangmanPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(26 + 20, 64),  # guessed letters + word state
            nn.ReLU(),
            nn.Linear(64, 26)
        )

    def forward(self, x):
        return self.net(x)

    def preprocess(self, state, guess_mask):
        # Convert '_pp_e' into a 20-char one-hot representation (or padded)
        vec = []
        for c in state:
            if c == "_":
                vec.extend([0]*26)
            else:
                one_hot = [0]*26
                one_hot[string.ascii_lowercase.index(c)] = 1
                vec.extend(one_hot)
        # Pad if word < 20 chars
        while len(vec) < 26*20:
            vec.extend([0]*26)

        return guess_mask + vec[:26*20]

