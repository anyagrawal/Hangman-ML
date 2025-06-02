from collections import Counter

def guess_function(masked_word, guessed_letters, dictionary):
    """Simple letter frequency guesser"""
    all_letters = "".join(dictionary)
    frequency_order = [l for l, _ in Counter(all_letters).most_common()]
    for letter in frequency_order:
        if letter not in guessed_letters:
            return letter
    return None
