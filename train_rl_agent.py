# train_rl_agent.py
import random
import string
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from simulator import play_game, load_dictionary
from trained_model import HangmanPolicy

model = HangmanPolicy()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

all_words = load_dictionary("words_250000_train.txt")

def simulate_episode(word, max_turns=10):
    guessed = set()
    state = "_" * len(word)
    history = []
    tries = max_turns

    while tries > 0 and "_" in state:
        mask = [1 if c in guessed else 0 for c in string.ascii_lowercase]
        x = model.preprocess(state, mask)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze()

        for i in range(26):
            if string.ascii_lowercase[i] in guessed:
                probs[i] = -1e9  # Mask guessed

        choice_index = torch.argmax(probs).item()
        guess_char = string.ascii_lowercase[choice_index]

        guessed.add(guess_char)
        new_state = "".join([c if c == guess_char or c in guessed else "_" for c in word])
        reward = sum([1 for a, b in zip(state, new_state) if a != b])  # +1 per revealed letter
        done = new_state == word

        history.append((x, choice_index, reward))
        state = new_state
        if guess_char not in word:
            tries -= 1

        if done:
            break

    return history

# Train loop
for epoch in tqdm(range(10000)):
    word = random.choice(all_words)
    episode = simulate_episode(word)

    for x, action, reward in episode:
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor([action])
        logits = model(x_tensor)
        loss = loss_fn(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "policy.pt")

