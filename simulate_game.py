# simulate_game.py
import random
from guess_agent import GuessAgent

class HangmanSimulator:
    def __init__(self, word_list_path="wordbank.txt", max_attempts=6):
        self.words = self.load_words(word_list_path)
        self.max_attempts = max_attempts
        self.agent = GuessAgent()

    def load_words(self, path):
        with open(path, 'r') as f:
            return [line.strip().lower() for line in f if len(line.strip()) <= 20 and line.strip().isalpha()]

    def play(self, secret):
        guessed = set()
        correct = ["-" for _ in secret]
        attempts = 0

        while attempts < self.max_attempts:
            pattern = "".join(correct)
            guess = self.agent.guess(pattern, guessed)

            if guess is None:
                break
            guessed.add(guess)

            if guess in secret:
                for i, c in enumerate(secret):
                    if c == guess:
                        correct[i] = guess
                if "-" not in correct:
                    return True
            else:
                attempts += 1
        return False

    def evaluate(self, n=1000):
        sample = random.sample(self.words, n)
        wins = 0
        for i, word in enumerate(sample):
            result = self.play(word)
            wins += int(result)
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{n}] Accuracy: {wins/(i+1):.2%}")
        print(f"Final accuracy over {n} games: {wins/n:.2%}")
        return wins / n

if __name__ == '__main__':
    sim = HangmanSimulator()
    sim.evaluate(n=1000)
