# guess_agent.py
import torch
import torch.nn.functional as F
import numpy as np
from trained_model import HangmanPolicy

class GuessAgent:
    def __init__(self, model_path='hangman_model.pt', device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HangmanPolicy().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

    def encode_pattern(self, pattern):
        # Convert dash (‘-’) to special token 26, known letters to index
        x = np.zeros((20, 27), dtype=np.float32)
        for i, c in enumerate(pattern.lower()):
            if i >= 20:
                break
            if c == '-':
                x[i][26] = 1.0  # unknown
            elif c in self.char_to_idx:
                x[i][self.char_to_idx[c]] = 1.0
        return x.flatten()

    def encode_guessed(self, guessed_letters):
        vec = np.zeros(26, dtype=np.float32)
        for c in guessed_letters:
            if c in self.char_to_idx:
                vec[self.char_to_idx[c]] = 1.0
        return vec

    def guess(self, pattern, guessed_letters):
        word_enc = self.encode_pattern(pattern)
        guessed_enc = self.encode_guessed(guessed_letters)
        input_vec = np.concatenate([word_enc, guessed_enc])[None, :]  # shape: (1, 566)
        input_tensor = torch.tensor(input_vec, device=self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Pick best unguessed letter
        best_letter = None
        best_score = -1
        for idx, prob in enumerate(probs):
            letter = self.alphabet[idx]
            if letter not in guessed_letters and prob > best_score:
                best_score = prob
                best_letter = letter
        return best_letter
