import torch
import random
import numpy as np
from game import SnakeAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.95 #discount rate 
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 255, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def get_state(self, game):
        head = game.snake[0]
        point_left = Point(head.x-20, head.y)
        point_right = Point(head.x+20, head.y)
        point_up = Point(head.x, head.y-20)
        point_down = Point(head.x, head.y+20)
        
        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN
        
        state = [
            #straight
            (direction_right and game.is_collision(point_right)) or
            (direction_left and game.is_collision(point_left)) or
            (direction_up and game.is_collision(point_up)) or
            (direction_down and game.is_collision(point_down)),
            
            #right
            (direction_up and game.is_collision(point_right)) or
            (direction_down and game.is_collision(point_left)) or
            (direction_left and game.is_collision(point_up)) or
            (direction_right and game.is_collision(point_down)),
            
            #left
            (direction_down and game.is_collision(point_right)) or
            (direction_up and game.is_collision(point_left)) or
            (direction_right and game.is_collision(point_up)) or
            (direction_left and game.is_collision(point_down)),
            
            #movement
            direction_left,
            direction_right,
            direction_up,
            direction_down,
            
            #location of food
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array (state, dtype=int)
        
    
    def stored_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample1 = random.sample(self.memory, BATCH_SIZE) #tuple
        else:
            sample1 = self.memory
            
        states, actions, rewards, next_states, dones = zip(*sample1)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        self.epsilon = 80 - self.n_games #more games = smaller epsilon
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon: #smaller epsilon = less frequent randomness
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move
            
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    luna = Agent()
    game = SnakeAI()
    while True:
        state_old = luna.get_state(game)
        final_move = luna.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = luna.get_state(game)
        luna.train_short_memory(state_old, final_move, reward, state_new, done)
        luna.stored_memory(state_old, final_move, reward, state_new, done)
        
        if done:
            game.reset()
            luna.n_games += 1
            luna.train_long_memory()
            
            if score > best_score:
                best_score = score
                luna.model.save()
                
            print('Game', luna.n_games, 'Score', score, 'Best Score:', best_score)
            
            #plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/luna.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()