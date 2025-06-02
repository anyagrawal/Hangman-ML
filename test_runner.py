from hangman_offline.py import HangmanSimulator
from guess_func_RL.py import guess_function

# Load dictionary
with open("words_250000_train.txt", "r") as f:
    dictionary = f.read().splitlines()

# Set up simulation
words_to_test = ["apple", "banana", "grapes", "cherry", "orange"]
wins = 0

sim = HangmanSimulator()

for word in words_to_test:
    sim.reset(word)
    while not sim.is_game_over():
        masked = sim.get_masked_word()
        guess = guess_function(masked, sim.guessed_letters, dictionary)
        sim.guess(guess)

    result = "WON" if sim.is_game_won() else "LOST"
    print(f"{word}: {result}")
    if sim.is_game_won():
        wins += 1

print(f"\nSuccess rate: {wins}/{len(words_to_test)} = {wins / len(words_to_test):.2f}")
