import torch

class NeuralAgent:
    def __init__(self, model, paddle):
        self.model = model
        self.paddle = paddle

    def get_action(self, state, encoded_state):  # Receive encoded state
        with torch.no_grad():
            action_probs = self.model(encoded_state)
            return torch.argmax(action_probs).item()