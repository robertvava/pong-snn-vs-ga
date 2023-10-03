from pongconfig import config
from env.pongenv import PongEnvironment
from agents.heuristic import HeuristicAgent
from gui import PongUI
from env.objects import *
from agents.neural import NeuralAgent
from nets.ga import GA
from nets.snn import SpikingNN
import matplotlib.pyplot as plt
import os 
import torch
from agents.dqnagent import DQNAgent

def play_game(mode="play", agent_a=None, agent_b=None, play_with_wall=True, generations=10, game_duration = 10000):

    paddle_a = Paddle(0, config.SCREEN_HEIGHT // 2 - config.PADDLE_HEIGHT // 2)

    if play_with_wall:
        paddle_b = Wall(config.SCREEN_WIDTH - config.PADDLE_WIDTH, 0, config.SCREEN_HEIGHT)
    else:
        paddle_b = Paddle(config.SCREEN_WIDTH - config.PADDLE_WIDTH, config.SCREEN_HEIGHT // 2 - config.PADDLE_HEIGHT // 2)

    env = PongEnvironment(paddle_a=paddle_a, paddle_b=paddle_b, game_duration = game_duration)
   
    if agent_a == "ga":
        if os.path.isfile('agents/best_nn_ga_misc.pt'):
            best_nn = torch.load('agents/best_nn_ga_misc.pt')
        else: 
            best_nn, avg_fitnesses, max_fitnesses, min_fitnesses = train_mode_ga(env, generations = generations, game_duration = game_duration)
            torch.save(best_nn, 'agents/best_nn_ga_misc.pt')
        agent_a = NeuralAgent(best_nn, env.paddle_a)

    elif agent_a == "snn":
        env.with_reward = True
        agent_a = DQNAgent(state_size=6, action_size=3, model=SpikingNN())
        cumulative_reward = 0
        
        for i in range(game_duration):
            
            state = env.get_state()
            encoded_state = env.encode_state(state)
            action_a = agent_a.act(encoded_state)
            next_state, reward, done = env.step(action_a, action_b=None)
            cumulative_reward += reward
            agent_a.rewards.append(cumulative_reward)
            # Store experience in agent's memory
            agent_a.remember(encoded_state, action_a, reward, next_state, done)
            # Train the agent periodically
            if len(agent_a.memory) > 32:  # Assuming batch size of 32
                agent_a.replay(32)

            if i % 10 == 0:
                print (f'States at action {i+1}: Reward: {cumulative_reward}')

        play_game(mode="play", play_with_wall = True, agent_a=agent_a, game_duration = 100000)
        plot_metrics_snn(agent_a, encoded_state)
        

    else:
        agent_a = HeuristicAgent(env.paddle_a)
    
    if mode == "play":
        play_mode(env, agent_a, agent_b, game_duration = game_duration)
    elif mode == "train":
        play_game(mode="play", agent_a=agent_a, game_duration = 1000000)

        plot_metrics_ga(avg_fitnesses, max_fitnesses, min_fitnesses)

def play_mode(env, agent_a, agent_b, game_duration = 10000):
    ui = PongUI(env)
    frame = 0
    running = True

    while running:
        frame += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        state = env.get_state()
        encoded_state = env.encode_state(state)
        action_a = agent_a.get_action(state, encoded_state)
        
        env.step(action_a, action_b=None)
        ui.render()


def train_mode_ga(env, generations = 10, game_duration= 10000):
    ga = GA(pop_size=20, mutation_rate = config.MUTATION_RATE, crossover_rate=config.CROSSOVER_RATE)
    avg_fitnesses = []
    max_fitnesses = []
    min_fitnesses = []

    for generation in range(generations):
        fitness_scores = [ga.evaluate_fitness(ind) for ind in ga.population]
        ga.evolve()
       
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        max_fitness = max(fitness_scores)
        min_fitness = min(fitness_scores)

        avg_fitnesses.append(avg_fitness)
        max_fitnesses.append(max_fitness)
        min_fitnesses.append(min_fitness)

        if generation % 5 == 0:
            plot_fitness_distribution(fitness_scores, generation + 1)
        
        print(f"Generation {generation + 1} - Average Fitness: {avg_fitness}")


    return ga.get_best_individual(), avg_fitnesses, max_fitnesses, min_fitnesses


def plot_metrics_ga(avg_fitnesses, max_fitnesses, min_fitnesses):
    generations = range(1, len(avg_fitnesses) + 1)

    plt.figure(figsize=(12, 6))

    plt.plot(generations, avg_fitnesses, label='Average Fitness', color='blue')
    plt.plot(generations, max_fitnesses, label='Max Fitness', color='green')
    plt.plot(generations, min_fitnesses, label='Min Fitness', color='red')

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('GA Fitness over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_fitness_distribution(fitness_scores, generation):
    plt.figure(figsize=(8, 5))
    plt.hist(fitness_scores, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Fitness')
    plt.ylabel('Number of Individuals')
    plt.title(f'Fitness Distribution - Generation {generation}')
    plt.grid(True)
    plt.show()

def plot_metrics_snn(agent, last_encoded_state):
    # Reward Over Time
    plt.plot(agent.rewards)
    plt.title('Reward Over Time')
    plt.xlabel('Games')
    plt.ylabel('Cumulative Reward')
    plt.show()

    # Epsilon Decay
    plt.plot(agent.epsilon_values)
    plt.title('Epsilon Decay Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Epsilon Value')
    plt.show()

    # Loss Over Time
    plt.plot(agent.losses)
    plt.title('Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.show()

    # Action Distribution
    actions = list(agent.actions_taken.keys())
    counts = list(agent.actions_taken.values())
    plt.bar(actions, counts)
    plt.title('Action Distribution')
    plt.xlabel('Actions')
    plt.ylabel('Count')
    plt.xticks(actions)
    plt.show()

    # Membrane Potential Visualization (for the last game)
    outputs = agent.model(last_encoded_state)
    for neuron in range(outputs.shape[1]):
        plt.plot(outputs[:, neuron].detach().numpy())
    plt.title('Membrane Potentials Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Membrane Potential')
    plt.show()

    # Q-Value Visualization for a sample state
    sample_state = torch.rand((1, 6))  # Random sample state
    q_values = agent.model(sample_state).squeeze().detach().numpy()
    plt.bar(range(len(q_values)), q_values)
    plt.title('Q-Values for Sample State')
    plt.xlabel('Actions')
    plt.ylabel('Q-Value')
    plt.show()



agent_a = "snn"
# agent_b = 
play_game(mode="train", agent_a = agent_a, play_with_wall = True, generations = 25, game_duration = 5000)

