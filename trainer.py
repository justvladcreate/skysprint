import math
import numpy as np
import torch
from env import SkySprintEnv
from agent import NeuroevoAgent
from config import (
    DEFAULT_POP_SIZE, DEFAULT_MAX_STEPS,
    DEFAULT_INPUT_SIZE, DEFAULT_HIDDEN_SIZES
)

class PopulationTrainer:
    def __init__(self, level_file,
                 pop_size=DEFAULT_POP_SIZE,
                 max_steps=DEFAULT_MAX_STEPS,
                 weak_mutation_rate=0.01,
                 weak_mutation_scale=0.01,
                 medium_mutation_rate=0.1,
                 medium_mutation_scale=0.2,
                 strong_mutation_scale=5.0,
                 weak_ratio=0.6,
                 medium_ratio=0.3,
                 input_size=DEFAULT_INPUT_SIZE,
                 hidden_sizes=DEFAULT_HIDDEN_SIZES,
                 init_weights=None,          # numpy-массив весов загруженной модели (может быть None)
                 device='cuda'):
        self.level_file = level_file
        self.pop_size = pop_size
        self.max_steps = max_steps

        self.weak_mutation_rate = weak_mutation_rate
        self.weak_mutation_scale = weak_mutation_scale
        self.medium_mutation_rate = medium_mutation_rate
        self.medium_mutation_scale = medium_mutation_scale
        self.strong_mutation_scale = strong_mutation_scale
        self.weak_ratio = weak_ratio
        self.medium_ratio = medium_ratio
        self.strong_ratio = 1.0 - weak_ratio - medium_ratio

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.env_template = SkySprintEnv(level_file, use_rays=True)
        self.envs = [SkySprintEnv(level_file, use_rays=True) for _ in range(pop_size)]
        self.agents = [NeuroevoAgent(input_size, hidden_sizes, self.device) for _ in range(pop_size)]

        self.states = [None] * pop_size
        self.dones = [False] * pop_size
        self.fitnesses = np.zeros(pop_size)
        self.last_cp = [0] * pop_size
        self.steps_since_last_cp = [0] * pop_size
        self.step_counter = 0
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_weights = None
        self.avg_fitness = 0.0

        self.best_time = None
        self.best_deaths = 0
        self.agent_step_counters = [0] * pop_size
        self.agent_deaths = [0] * pop_size

        # Если переданы начальные веса, инициализируем популяцию особым образом
        if init_weights is not None:
            self._init_from_weights(init_weights)
        else:
            self._reset_generation()

    def _init_from_weights(self, weights):
        """Создаёт популяцию на основе загруженной модели.
           Первый агент – точная копия, остальные – её мутанты."""
        # Установим веса для всех агентов, начиная с копии и мутантов
        base_agent = NeuroevoAgent(self.input_size, self.hidden_sizes, self.device)
        base_agent.set_weights_flat(weights)
        self.agents[0] = base_agent.copy()
        for i in range(1, self.pop_size):
            mutant = base_agent.copy()
            # Применяем слабую мутацию, чтобы создать разнообразие
            NeuroevoAgent.mutate(mutant, mode='weak',
                                 mutation_rate=self.weak_mutation_rate,
                                 mutation_scale=self.weak_mutation_scale)
            self.agents[i] = mutant

        # Сбрасываем поколение с этими агентами
        self._reset_generation()

    def _reset_generation(self):
        for i, env in enumerate(self.envs):
            env.reset()
            self.states[i] = env.get_state(1)
        self.dones = [False] * self.pop_size
        self.fitnesses.fill(0.0)
        self.last_cp = [0] * self.pop_size
        self.steps_since_last_cp = [0] * self.pop_size
        self.step_counter = 0
        self.agent_step_counters = [0] * self.pop_size
        self.agent_deaths = [0] * self.pop_size

    def step_all(self):
        alive_mask = [not d for d in self.dones]
        if not any(alive_mask):
            return False

        alive_indices = [i for i, a in enumerate(alive_mask) if a]
        state_tensors = []
        for idx in alive_indices:
            tensor = self._state_to_tensor(self.states[idx], self.device)
            state_tensors.append(tensor)
        states_batch = torch.stack(state_tensors)

        actions = []
        for i, idx in enumerate(alive_indices):
            agent = self.agents[idx]
            with torch.no_grad():
                prob = agent.net(states_batch[i].unsqueeze(0)).item()
            actions.append(1 if prob > 0.5 else 0)

        for i, idx in enumerate(alive_indices):
            env = self.envs[idx]
            state = self.states[idx]
            next_state, _, done, _ = env.step(actions[i])

            self.agent_step_counters[idx] += 1

            if next_state['next_cp'] > self.last_cp[idx]:
                self.last_cp[idx] = next_state['next_cp']
                self.steps_since_last_cp[idx] = 0
            else:
                self.steps_since_last_cp[idx] += 1

            if not next_state['alive'] and state['alive']:
                self.agent_deaths[idx] += 1

            reward = self._calculate_reward(state, next_state, done, idx)
            self.fitnesses[idx] += reward
            self.states[idx] = next_state

            if done:
                self.dones[idx] = True
                if (self.step_counter >= self.max_steps - 1 and
                        not next_state['alive'] and
                        next_state['next_cp'] < len(env.checkpoints)):
                    self.fitnesses[idx] -= 100.0

        self.step_counter += 1
        return True

    def _state_to_tensor(self, state, device):
        cp_x = state['cp_x'] if state['cp_x'] is not None else 0
        cp_y = state['cp_y'] if state['cp_y'] is not None else 0
        rel_x = (cp_x - state['x']) / 2000.0
        rel_y = (cp_y - state['y']) / 1000.0
        vx = state['vx'] / 10.0
        vy = state['vy'] / 10.0
        angle = state['angle'] / math.pi
        rays = state.get('rays', [0.5]*5)
        features = [rel_x, rel_y, vx, vy, angle] + rays + [1.0]
        return torch.tensor(features[:self.input_size], dtype=torch.float32, device=device)

    def _calculate_reward(self, state, next_state, done, agent_idx):
        reward = -0.1
        if state['cp_x'] is not None and next_state['cp_x'] is not None:
            prev_dist = math.hypot(state['cp_x'] - state['x'], state['cp_y'] - state['y'])
            new_dist = math.hypot(next_state['cp_x'] - next_state['x'], next_state['cp_y'] - next_state['y'])
            reward += 0.05 * (prev_dist - new_dist)

        if next_state['next_cp'] > state['next_cp']:
            speed_bonus = max(0, 50 - self.steps_since_last_cp[agent_idx])
            reward += 100.0 + speed_bonus

        if done and next_state['next_cp'] >= len(self.envs[agent_idx].checkpoints):
            final_bonus = max(0, 100 - self.steps_since_last_cp[agent_idx] * 2)
            reward += 500.0 + final_bonus
        elif done and not next_state['alive']:
            reward -= 50.0

        return reward

    def generation_complete(self):
        return all(self.dones) or self.step_counter >= self.max_steps

    def evolve(self):
        best_idx = np.argmax(self.fitnesses)
        if self.fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = self.fitnesses[best_idx]
            self.best_weights = self.agents[best_idx].get_weights_flat().copy()
            self.best_time = self.agent_step_counters[best_idx] / 60.0
            self.best_deaths = self.agent_deaths[best_idx]

        self.avg_fitness = np.mean(self.fitnesses)

        sorted_indices = np.argsort(self.fitnesses)[::-1]
        sorted_agents = [self.agents[i] for i in sorted_indices]

        worst_count = max(1, int(self.pop_size * 0.1))
        good_count = self.pop_size - worst_count

        quotas = np.zeros(good_count, dtype=int)
        quotas[0] = max(1, int(self.pop_size * 0.1))
        if good_count > 1:
            quotas[1] = max(1, int(self.pop_size * 0.05))
        for i in range(2, min(10, good_count)):
            quotas[i] = max(1, int(self.pop_size * 0.03))

        total_allocated = np.sum(quotas)
        remaining = good_count - total_allocated

        if good_count > 10 and remaining > 0:
            other_parents = good_count - 10
            base = remaining // other_parents
            extra = remaining % other_parents
            for i in range(10, good_count):
                quotas[i] = base + (1 if i - 10 < extra else 0)
        elif good_count <= 10 and remaining > 0:
            quotas[0] += remaining

        new_agents = []
        for parent_idx in range(good_count):
            parent = sorted_agents[parent_idx]
            num_children = quotas[parent_idx]
            for _ in range(num_children):
                child = parent.copy()
                r = np.random.random()
                if r < self.strong_ratio:
                    mode = 'strong'
                    NeuroevoAgent.mutate(child, mode='strong', strong_scale=self.strong_mutation_scale)
                elif r < self.strong_ratio + self.medium_ratio:
                    mode = 'medium'
                    NeuroevoAgent.mutate(child, mode='medium',
                                         mutation_rate=self.medium_mutation_rate,
                                         mutation_scale=self.medium_mutation_scale)
                else:
                    mode = 'weak'
                    NeuroevoAgent.mutate(child, mode='weak',
                                         mutation_rate=self.weak_mutation_rate,
                                         mutation_scale=self.weak_mutation_scale)
                new_agents.append(child)

        boundary_agent = sorted_agents[good_count - 1]
        for _ in range(worst_count):
            new_agents.append(boundary_agent.copy())

        self.agents = new_agents
        self.generation += 1
        self._reset_generation()

    def get_best_agent(self):
        if self.best_weights is not None:
            best = NeuroevoAgent(self.input_size, self.hidden_sizes, 'cpu')
            best.set_weights_flat(self.best_weights)
            return best
        best_idx = np.argmax(self.fitnesses) if any(not d for d in self.dones) else np.argmax(self.fitnesses)
        best_agent = self.agents[best_idx].copy()
        best_agent.device = torch.device('cpu')
        best_agent.net.to('cpu')
        return best_agent

    def get_best_info(self):
        agent = self.get_best_agent()
        time = self.best_time if self.best_time is not None else 0.0
        deaths = self.best_deaths
        return agent, time, deaths

    def get_best_agent_index(self):
        if all(self.dones):
            return np.argmax(self.fitnesses)
        alive_fitness = [self.fitnesses[i] if not self.dones[i] else -float('inf') for i in range(self.pop_size)]
        return np.argmax(alive_fitness)

    def get_best_state(self):
        idx = self.get_best_agent_index()
        return self.states[idx]