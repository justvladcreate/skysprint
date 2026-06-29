import numpy as np
import torch
import torch.nn as nn
import math

class NeuroevoNet(nn.Module):
    def __init__(self, input_size=11, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

class NeuroevoAgent:
    def __init__(self, input_size=11, hidden_sizes=(64, 64), device='cpu'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.device = torch.device(device)
        self.net = NeuroevoNet(input_size, hidden_sizes).to(self.device)
        self.net.eval()
        # Дополнительные атрибуты для хранения статистики
        self.pb_time = None   # лучшее время прохождения (секунды)
        self.deaths = 0       # количество смертей за лучшую попытку

    def _state_to_tensor(self, state):
        cp_x = state['cp_x'] if state['cp_x'] is not None else 0
        cp_y = state['cp_y'] if state['cp_y'] is not None else 0
        rel_x = (cp_x - state['x']) / 2000.0
        rel_y = (cp_y - state['y']) / 1000.0
        vx = state['vx'] / 10.0
        vy = state['vy'] / 10.0
        angle = state['angle'] / math.pi
        rays = state.get('rays', [0.5]*5)
        features = [rel_x, rel_y, vx, vy, angle] + rays + [1.0]
        return torch.tensor(features[:self.input_size], dtype=torch.float32, device=self.device)

    def predict(self, state, epsilon=0.0):
        with torch.no_grad():
            x = self._state_to_tensor(state).unsqueeze(0)
            prob = self.net(x).item()
            if epsilon > 0 and np.random.random() < epsilon:
                return np.random.randint(0, 2)  # случайное действие
            return 1 if prob > 0.5 else 0

    def predict_from_tensor(self, state_tensor, epsilon=0.0):
        with torch.no_grad():
            prob = self.net(state_tensor).item()
            if epsilon > 0 and np.random.random() < epsilon:
                return np.random.randint(0, 2)
            return 1 if prob > 0.5 else 0

    def get_weights_flat(self):
        params = []
        for param in self.net.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_weights_flat(self, weights):
        idx = 0
        for param in self.net.parameters():
            shape = param.data.shape
            size = np.prod(shape)
            param.data = torch.from_numpy(weights[idx:idx+size].reshape(shape)).float().to(self.device)
            idx += size

    def copy(self):
        new_agent = NeuroevoAgent(self.input_size, self.hidden_sizes, self.device)
        new_agent.set_weights_flat(self.get_weights_flat())
        return new_agent

    @staticmethod
    def mutate(agent, mode='weak', **kwargs):
        weights = agent.get_weights_flat()
        if mode == 'strong':
            scale = kwargs.get('strong_scale', 5.0)
            idx = np.random.randint(0, len(weights))
            weights[idx] += np.random.randn() * scale
        else:
            rate = kwargs.get('mutation_rate', 0.01)
            scale = kwargs.get('mutation_scale', 0.01)
            noise = np.random.randn(len(weights)) * scale
            mask = np.random.random(len(weights)) < rate
            weights += noise * mask
        agent.set_weights_flat(weights)

    @staticmethod
    def crossover(parent1, parent2):
        w1 = parent1.get_weights_flat()
        w2 = parent2.get_weights_flat()
        child_weights = (w1 + w2) / 2.0
        child = NeuroevoAgent(parent1.input_size, parent1.hidden_sizes, parent1.device)
        child.set_weights_flat(child_weights)
        return child

    def save(self, filepath, pb_time=None, deaths=None):
        """
        Сохраняет модель, опционально добавляя лучшее время и смерти.
        """
        data = {
            'weights': self.get_weights_flat(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes
        }
        if pb_time is not None:
            data['pb_time'] = pb_time
        if deaths is not None:
            data['deaths'] = deaths
        torch.save(data, filepath)

    @classmethod
    def load(cls, filepath, device='cpu'):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'weights' in checkpoint:
            input_size = checkpoint.get('input_size', 11)
            hidden_sizes = checkpoint.get('hidden_sizes', (64, 64))
            weights = checkpoint['weights']
            agent = cls(input_size, hidden_sizes, device)
            agent.set_weights_flat(weights)
            # Восстанавливаем статистику, если есть
            agent.pb_time = checkpoint.get('pb_time', None)
            agent.deaths = checkpoint.get('deaths', 0)
            return agent
        else:
            try:
                if isinstance(checkpoint, torch.Tensor):
                    weights = checkpoint.cpu().numpy().flatten()
                elif isinstance(checkpoint, np.ndarray):
                    weights = checkpoint.flatten()
                else:
                    weights = np.array(checkpoint).flatten()
            except Exception as e:
                raise ValueError(f"Unsupported model format: {e}")
            agent = cls(device=device)
            agent.set_weights_flat(weights)
            return agent