# train_rl_agent.py
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from guess_agent import GuessAgent
from trained_model import HangmanPolicy

class RLSelfPlayTrainer:
    def __init__(self, wordbank_path="wordbank.txt", model_path="hangman_model.pt", episodes=10000):
        self.words = self.load_words(wordbank_path)
        self.model_path = model_path
        self.episodes = episodes
        self.model = HangmanPolicy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def load_words(self, path):
        with open(path, 'r') as f:
            return [line.strip().lower() for line in f if line.strip().isalpha() and len(line.strip()) <= 20]

    def encode(self, pattern, guessed):
        word_vec = [(ord(c) - 97 + 1) if c != '-' else 0 for c in pattern]
        word_vec += [0] * (20 - len(word_vec))
        guessed_vec = [1 if chr(i + 97) in guessed else 0 for i in range(26)]
        return torch.tensor(word_vec + guessed_vec, dtype=torch.float32)

    def select_action(self, output, guessed):
        probs = output.detach().clone()
        for i in range(26):
            if chr(i + 97) in guessed:
                probs[i] = -1e9
        return torch.argmax(probs).item()

    def simulate_episode(self, secret):
        guessed = set()
        correct = ["-" for _ in secret]
        states, targets = [], []

        for _ in range(6):
            pattern = "".join(correct)
            x = self.encode(pattern, guessed)
            output = self.model(x)
            guess_idx = self.select_action(output, guessed)
            guess = chr(guess_idx + 97)
            guessed.add(guess)
            states.append(x)
            targets.append(torch.tensor([guess_idx]))

            if guess in secret:
                for i, c in enumerate(secret):
                    if c == guess:
                        correct[i] = guess
                if "-" not in correct:
                    return states, targets, True
        return states, targets, False

    def train(self):
        for _ in tqdm(range(self.episodes)):
            word = random.choice(self.words)
            states, targets, win = self.simulate_episode(word)
            reward = 1.0 if win else -0.1

            for x, target in zip(states, targets):
                pred = self.model(x)
                loss = self.criterion(pred.unsqueeze(0), target)
                self.optimizer.zero_grad()
                loss.backward()
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad *= reward
                self.optimizer.step()

        torch.save(self.model.state_dict(), self.model_path)
        print(f"RL training complete. Model saved to {self.model_path}.")

if __name__ == '__main__':
    trainer = RLSelfPlayTrainer()
    trainer.train()