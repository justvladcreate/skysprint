import os
import time
import copy
import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from env import SkySprintEnv
from agent_dqn import DQNAgent

# Для безопасности при использовании CUDA и multiprocessing
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

def worker_episode(level_path, weights_path, epsilon, max_steps,
                   input_size=11, hidden_sizes=(64,64), seed=None):
    """Собирает один эпизод, теперь с бонусом за приближение к чекпоинту."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    env = SkySprintEnv(level_path, use_rays=True)
    agent = DQNAgent(input_size, hidden_sizes, device='cpu')
    agent.q_network.load_state_dict(torch.load(weights_path, map_location='cpu'))
    agent.q_network.eval()

    state = env.get_state(1)
    trajectory = []
    total_reward = 0.0
    steps = 0
    total_checkpoints = len(env.checkpoints)
    last_cp = 0
    steps_since_last_cp = 0

    # Функция для вычисления расстояния до следующего чекпоинта
    def distance_to_next(state):
        if state is None or state['cp_x'] is None:
            return 0.0
        dx = state['cp_x'] - state['x']
        dy = state['cp_y'] - state['y']
        return math.hypot(dx, dy)

    prev_dist = distance_to_next(state)

    while steps < max_steps:
        action = agent.act(state, epsilon)
        next_state, _, done, _ = env.step(action)
        steps_since_last_cp += 1

        # Базовая награда (штраф за время)
        reward = -0.1

        # Бонус за приближение к чекпоинту
        if not done and next_state['cp_x'] is not None:
            new_dist = distance_to_next(next_state)
            # Награда пропорциональна уменьшению расстояния (положительная, если стали ближе)
            reward += 0.05 * (prev_dist - new_dist)   # маленький коэффициент, чтобы не перебивать основные награды
            prev_dist = new_dist
        else:
            prev_dist = 0.0  # если done, неважно

        # Прогресс по чекпоинтам
        cp = next_state['next_cp']
        if cp > last_cp:
            speed_bonus = max(0, 50 - steps_since_last_cp)
            reward += 100.0 + speed_bonus
            last_cp = cp
            steps_since_last_cp = 0
            # После взятия кольца сбрасываем предыдущее расстояние (оно обновится на следующем шаге)
            prev_dist = distance_to_next(next_state)

        if done and cp >= total_checkpoints:
            final_bonus = max(0, 100 - steps_since_last_cp * 2)
            reward += 500.0 + final_bonus
        elif done:
            reward -= 50.0

        trajectory.append((state, action, reward, next_state, done))
        total_reward += reward
        state = next_state
        if done:
            break
        steps += 1

    if not done:
        total_reward -= 100.0
    return trajectory, total_reward


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    def __init__(self, level_file,
                 num_agents=20, episodes=200, max_steps=2000,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                 gamma=0.99, lr=1e-3, batch_size=256, buffer_capacity=100000,
                 target_update_freq=50,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.level_file = level_file
        self.num_agents = num_agents
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.agent = DQNAgent(device=self.device)
        self.optimizer = optim.Adam(self.agent.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.progress = 0.0
        self.best_reward = -float('inf')
        self.best_model_state = None
        self.stop_requested = False
        self.current_episode = 0

    def _update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        state_tensors = torch.stack([self.agent._state_to_tensor(s) for s in states])
        next_state_tensors = torch.stack([self.agent._state_to_tensor(s) for s in next_states])
        action_tensors = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_tensors = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_tensors = torch.tensor(dones, dtype=torch.bool, device=self.device)

        q_values = self.agent.q_network(state_tensors)
        q_values = q_values.gather(1, action_tensors.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.agent.target_network(next_state_tensors)
            max_next_q = next_q_values.max(1)[0]
            target = reward_tensors + self.gamma * max_next_q * (~done_tensors)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self, pool, progress_callback=None):
        weights_path = f"temp_weights_{id(self)}.pth"
        for ep in range(self.episodes):
            if self.stop_requested:
                break
            self.current_episode = ep + 1

            # Сохраняем текущие веса для воркеров
            torch.save(self.agent.q_network.state_dict(), weights_path)

            # Параллельный сбор опыта (воркеры на CPU)
            args = [(self.level_file, weights_path, self.epsilon,
                     self.max_steps, 11, (64, 64),
                     np.random.randint(0, 2**31)) for _ in range(self.num_agents)]
            results = pool.starmap(worker_episode, args)

            ep_rewards = []
            for trajectory, total_r in results:
                ep_rewards.append(total_r)
                for s, a, r, ns, done in trajectory:
                    self.replay_buffer.push(s, a, r, ns, done)

            avg_reward = np.mean(ep_rewards) if ep_rewards else 0.0
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.best_model_state = copy.deepcopy(self.agent.q_network.state_dict())

            # Интенсивное обучение на GPU: много шагов с большим батчем
            if len(self.replay_buffer) >= self.batch_size:
                for _ in range(20):  # 20 итераций обучения на эпизод
                    self._train_step()

            if ep % self.target_update_freq == 0:
                self.agent.update_target()

            self._update_epsilon()
            self.progress = (ep + 1) / self.episodes

            if progress_callback:
                progress_callback(self.progress, ep + 1, self.episodes,
                                  self.best_reward, avg_reward, self.epsilon)

        if os.path.exists(weights_path):
            os.remove(weights_path)

    def save_best(self, path):
        if self.best_model_state is not None:
            best_agent = DQNAgent(device='cpu')
            best_agent.q_network.load_state_dict(self.best_model_state)
            best_agent.save(path)
        else:
            self.agent.save(path)