# guess_function.py
import string
import numpy as np
import torch
from trained_model import HangmanPolicy

# Load trained model (you’ll generate this later)
model = HangmanPolicy()
model.load_state_dict(torch.load("policy.pt"))
model.eval()

guessed_letters = set()

def guess(word):
    # convert "_ p p _ e" -> "_pp_e"
    obs = word[::2]

    # Build 26-dim one-hot for guessed letters
    guess_mask = [1 if letter in guessed_letters else 0 for letter in string.ascii_lowercase]

    # Create feature: masked word + guessed letters
    input_vector = model.preprocess(obs, guess_mask)
    input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        scores = model(input_tensor).squeeze().numpy()

    # Mask out already guessed letters
    for i, letter in enumerate(string.ascii_lowercase):
        if letter in guessed_letters:
            scores[i] = -np.inf

    choice = string.ascii_lowercase[np.argmax(scores)]
    guessed_letters.add(choice)
    return choice
