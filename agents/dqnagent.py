import random
from collections import deque
import torch

class DQNAgent:
    def __init__(self, state_size, action_size, model):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    
        self.epsilon = 0.80 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = torch.nn.MSELoss()

      
        self.rewards = []
        self.losses = []
        self.actions_taken = {0: 0, 1: 0, 2: 0}
        self.epsilon_values = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            
            self.epsilon_values.append(self.epsilon)

            return random.randrange(self.action_size)
        
        self.epsilon_values.append(self.epsilon)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()  

        

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state).squeeze()).item()
            target_f = self.model(state).squeeze()
            target_f[action] = target
            self.optimizer.zero_grad()
            outputs = self.model(state).squeeze()
            loss = self.criterion(outputs, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay