import torch
import numpy as np
import random
from collections import deque
import cv2
import pygame
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot_learning_curve
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

BLOCK_SIZE = 20
MAX_MEMORY = 100000
BATCH_SIZE = 500
LR = 1e-4
VIDEO_FPS = 30
VIDEO_PATH = "training.mp4"
RECORD_START_GAME = 400
REPLAY_START = 1000

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 
        self.gamma = 0.9
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay_games = 1000
        self. memory = deque(maxlen = MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)
        
    def get_state(self, game):
        head = game.head
        
        #check border
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)   
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        #direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        #check danger
        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)),

            (dir_d and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_d)),
            
            dir_l, dir_r, dir_u, dir_d,
            
            #check food
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
            ]
        
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)
        if not mini_sample:
            return
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        self.epsilon = max(0.01, 1 - self.n_games / 200)
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randrange(3)
        else:
            state0 = torch.as_tensor(state, dtype=torch.float32)
            with torch.no_grad():
                q = self.model(state0).flatten()
            move = int(torch.argmax(q[:3]).item()) if q.numel() >= 3 and torch.isfinite(q[:3]).all() else random.randrange(3)
        final_move[move] = 1
        return final_move

def make_plot_image(scores, height):
    fig = Figure(figsize=(4, height/100), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_title("Scores")
    ax.set_xlabel("Game")
    ax.set_ylabel("Score")
    if scores:
        ax.plot(range(1, len(scores)+1), scores, color="blue")
        ax.set_ylim(ymin=0)
    ax.grid(True)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    h, w = canvas.get_width_height()[1], canvas.get_width_height()[0]
    img = buf.reshape((h, w, 4))[:, :, :3]
    return img

def train():
    plot_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    game.reset()
    writer = None

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
            print('Game', agent.n_games, 'Score', score, 'Record', record)
            plot_scores.append(score)

        if agent.n_games >= RECORD_START_GAME:
            surf = pygame.display.get_surface()
            if surf is not None and writer is not None:
                game_surface = pygame.surfarray.array3d(surf)
                game_surface = np.transpose(game_surface, (1, 0, 2))[:, :, ::-1]
                plot_img = make_plot_image(plot_scores, game_surface.shape[0])
                plot_img = cv2.resize(plot_img, (400, game_surface.shape[0]))
                combined = cv2.hconcat([game_surface, plot_img])
                writer.write(combined)

            if writer is None:
                surf = pygame.display.get_surface()
                if surf is None:
                    continue
                gw, gh = surf.get_size()
                pw = 400
                video_path_local = VIDEO_PATH
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(video_path_local, fourcc, VIDEO_FPS, (gw+pw, gh))
                if not writer.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_path_local = "training.avi"
                    writer = cv2.VideoWriter(video_path_local, fourcc, VIDEO_FPS, (gw+pw, gh))
                    if not writer.isOpened():
                        raise RuntimeError("OpenCV VideoWriter failed with mp4v and XVID")


    if writer is not None:
        writer.release()

if __name__ == '__main__':
    train()