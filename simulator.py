import random
import re
from collections import Counter

class LocalHangmanSimulator:
    def __init__(self, word_list, max_attempts=6):
        self.word_list = word_list
        self.max_attempts = max_attempts

    def reset(self, secret_word=None):
        self.secret_word = secret_word or random.choice(self.word_list)
        self.guessed_letters = set()
        self.remaining_attempts = self.max_attempts
        self.visible_word = ['_' for _ in self.secret_word]
        return ''.join(self.visible_word)

    def get_masked_word(self):
        return ' '.join(self.visible_word)

    def guess(self, letter):
        if letter in self.guessed_letters:
            return False, self.get_masked_word(), self.remaining_attempts, False
        
        self.guessed_letters.add(letter)

        if letter in self.secret_word:
            for i, c in enumerate(self.secret_word):
                if c == letter:
                    self.visible_word[i] = letter
            success = '_' not in self.visible_word
            return True, self.get_masked_word(), self.remaining_attempts, success
        else:
            self.remaining_attempts -= 1
            return False, self.get_masked_word(), self.remaining_attempts, False

    def play_game(self, guess_function):
        self.reset()
        success = False

        while self.remaining_attempts > 0 and '_' in self.visible_word:
            masked_word = self.get_masked_word()
            guess = guess_function(masked_word, list(self.guessed_letters))
            correct, new_masked, remaining, success = self.guess(guess)
            if success:
                break

        return {
            'word': self.secret_word,
            'success': success,
            'remaining_attempts': self.remaining_attempts,
            'guessed_letters': list(self.guessed_letters)
        }

# Example dictionary
sample_word_list = ["apple", "orange", "banana", "grape", "melon", "kiwi"]

simulator = LocalHangmanSimulator(sample_word_list)
simulator.reset("apple")
simulator.get_masked_word()  # Output initial state for verification

