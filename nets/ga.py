
from pongconfig import config
from nets.net import SimpleNN
import random
import torch
import numpy as np
from env.pongenv import PongEnvironment
import math

class GA:
    def __init__(self, pop_size, mutation_rate, crossover_rate):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = [SimpleNN() for _ in range(pop_size)]

    def select_parents(self):
        parent1 = self.tournament_selection()
        parent2 = self.tournament_selection()
        while parent1 == parent2: 
            parent2 = self.tournament_selection()
        return parent1, parent2
    
    def evaluate_fitness(self, individual):
        env = PongEnvironment()
        done = False
        hits = 0

        while not done:
            state = env.get_state()
            encoded_state = env.encode_state(state)
            action_probs = individual(encoded_state)
            action = torch.argmax(action_probs).item()
            _, done = env.step(action, action_b=None)
            
            if env.ball.hit_paddle_a:
                hits += 1

        R_hit = 1  # Reward for hitting the ball
        P_score = 2  # Penalty for letting the wall score
        fitness = (R_hit * hits) - (P_score * env.game.score_b)
        # fitness = fitness / 100
        return 100 + fitness

    def crossover(self, parent1, parent2):
        child = SimpleNN()
       
        mask = torch.rand_like(parent1.fc1.weight) < self.crossover_rate
        inverted_mask = ~mask  

        child.fc1.weight.data = mask.float() * parent1.fc1.weight.data + inverted_mask.float() * parent2.fc1.weight.data
        mask = torch.rand_like(parent1.fc2.weight) < self.crossover_rate
        inverted_mask = ~mask  

        child.fc2.weight.data = mask.float() * parent1.fc2.weight.data + inverted_mask.float() * parent2.fc2.weight.data
        mask = torch.rand_like(parent1.fc3.weight) < self.crossover_rate
        inverted_mask = ~mask 

        child.fc3.weight.data = mask.float() * parent1.fc3.weight.data + inverted_mask.float() * parent2.fc3.weight.data
        mask = torch.rand_like(parent1.fc4.weight) < self.crossover_rate
        inverted_mask = ~mask  

        child.fc4.weight.data = mask.float() * parent1.fc4.weight.data + inverted_mask.float() * parent2.fc4.weight.data

        return child

    def mutate(self, child):

        mutation = torch.randn_like(child.fc1.weight) * self.mutation_rate
        child.fc1.weight.data += mutation
        mutation = torch.randn_like(child.fc2.weight) * self.mutation_rate
        child.fc2.weight.data += mutation
        mutation = torch.randn_like(child.fc3.weight) * self.mutation_rate
        child.fc3.weight.data += mutation
        mutation = torch.randn_like(child.fc4.weight) * self.mutation_rate
        child.fc4.weight.data += mutation

    def tournament_selection(self, k=10):
        """Select a parent using tournament selection. Slect k random individuals."""
        selected = random.sample(self.population, k)
        fitness_scores = [self.evaluate_fitness(ind) for ind in selected]
        return selected[np.argmax(fitness_scores)]
    
    def evolve(self):
        fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
        parents = np.argsort(fitness_scores)[-2:]                                   # Select the top 2 individuals!!!
        parent1, parent2 = self.population[parents[0]], self.population[parents[1]]
        
        new_population = []
        for _ in range(self.pop_size):
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        self.population = new_population
    
    def get_best_individual(self):
        fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
        best_index = np.argmax(fitness_scores)
        return self.population[best_index]


# print ()
# mask1 = torch.rand_like(parent1.fc1.weight) < self.crossover_rate
# print (mask1)
# print (mask1.dtype)
# print (parent1.fc1.weight.data + (1 - torch.rand_like(parent1.fc1.weight) < self.crossover_rate) * parent2.fc1.weight.data)
# print (parent1.fc1.weight.data)
# mask = torch.rand_like(parent1.fc2.weight) < self.crossover_rate
# print (mask)
# print (parent1.fc2.weight.data)
# print(mask.shape, mask.dtype)
# print(parent1.fc2.weight.shape, parent1.fc2.weight.dtype)
# print(parent2.fc2.weight.shape, parent2.fc2.weight.dtype)



# def evaluate_fitness(self, individual):
#     env = PongEnvironment()
#     done = False
#     fitness = 0
#     prev_state = None

#     while not done:
#         state = env.get_state()
#         state_tensor = env.encode_state(state, prev_state)
        
#         action_probs = individual(state_tensor)
#         action = torch.argmax(action_probs).item()

#         prev_state = state.copy()
#         _, done = env.step(action, action_b=None)

#         action_probs = individual(state_tensor)
#         action = torch.argmax(action_probs).item()

#         prev_ball_y = state["ball_y"]
#         _, done = env.step(action, action_b=None)
#         new_ball_y = env.get_state()["ball_y"]

#         # Reward for ball tracking
#         if abs(state["paddle_a_y"] - state["ball_y"]) < config.PADDLE_HEIGHT:
#             fitness += 0.1

#         # Reward for anticipation
#         if (new_ball_y > prev_ball_y and action == 1) or (new_ball_y < prev_ball_y and action == 0):
#             fitness += 0.1

#         # Penalty for unnecessary movement
#         if abs(state["ball_speed_y"]) < 0.5 and action != 2:  # Assuming action 2 is staying still
#             fitness -= 0.05

#         # Reward for hitting the ball
#         if env.ball.hit_paddle:
#             fitness += 1

#         # Reward for defensive positioning
#         if state["ball_speed_y"] < 0 and abs(state["paddle_a_y"] - config.SCREEN_HEIGHT // 2) < config.PADDLE_HEIGHT:
#             fitness += 0.1

#     # Penalty for letting the wall score
#     fitness -= 5 * env.game.score_b

#     return fitness
