from __future__ import annotations

import json
import math
import pickle
import random
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from sim.port_env import PortEnvironment, State


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "port_optimus_q_agent.pkl"


@dataclass
class AgentConfig:
    learning_rate: float = 0.12
    discount_factor: float = 0.94
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.992
    episodes: int = 280
    action_size: int = 3


class GranularQLearningAgent:
    """Q-learning agent with granular state discretization for stable demos."""

    def __init__(self, config: Optional[AgentConfig] = None, seed: Optional[int] = 42) -> None:
        self.config = config or AgentConfig()
        self.rng = random.Random(seed)
        self.q_table: Dict[Tuple[int, int, int, int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(self.config.action_size, dtype=np.float32)
        )
        self.training_rewards: List[float] = []

    def discretize(self, state: State) -> Tuple[int, int, int, int, int]:
        queue, price, battery, trucks, hour = state
        queue_bin = int(np.digitize(queue, [5, 10, 16, 24, 34, 45]))
        price_bin = int(np.digitize(price, [0.18, 0.26, 0.38, 0.52, 0.68]))
        battery_bin = int(np.digitize(battery, [15, 30, 45, 60, 75, 90]))
        truck_bin = int(np.digitize(trucks, [3, 5, 7, 9]))
        hour_bin = int(np.digitize(hour % 24, [5, 9, 13, 17, 21]))
        return queue_bin, price_bin, battery_bin, truck_bin, hour_bin

    def choose_action(self, state: State, training: bool = True) -> int:
        key = self.discretize(state)
        if training and self.rng.random() < self.config.epsilon:
            return self.rng.randrange(self.config.action_size)
        return int(np.argmax(self.q_table[key]))

    def learn(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        key = self.discretize(state)
        next_key = self.discretize(next_state)
        old_value = self.q_table[key][action]
        next_value = 0.0 if done else float(np.max(self.q_table[next_key]))
        target = reward + self.config.discount_factor * next_value
        self.q_table[key][action] = old_value + self.config.learning_rate * (target - old_value)

    def decay_epsilon(self) -> None:
        self.config.epsilon = max(
            self.config.epsilon_min,
            self.config.epsilon * self.config.epsilon_decay,
        )

    def train(self, env_factory: Callable[[], PortEnvironment], episodes: Optional[int] = None) -> List[float]:
        total_episodes = episodes or self.config.episodes
        for _ in range(total_episodes):
            env = env_factory()
            state = env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action = self.choose_action(state, training=True)
                next_state, reward, done, _ = env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
            self.training_rewards.append(round(episode_reward, 4))
            self.decay_epsilon()
        return self.training_rewards

    def predict(self, state: State) -> int:
        return self.choose_action(state, training=False)

    def save_model(self, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "q_table": {json.dumps(key): value.tolist() for key, value in self.q_table.items()},
            "training_rewards": self.training_rewards,
        }
        with path.open("wb") as file:
            pickle.dump(payload, file)
        return path

    @classmethod
    def load_model(cls, path: Path | str = DEFAULT_MODEL_PATH) -> "GranularQLearningAgent":
        path = Path(path)
        with path.open("rb") as file:
            payload = pickle.load(file)
        agent = cls(AgentConfig(**payload["config"]))
        for key, value in payload["q_table"].items():
            agent.q_table[tuple(json.loads(key))] = np.array(value, dtype=np.float32)
        agent.training_rewards = payload.get("training_rewards", [])
        return agent


def evaluate_agent(
    agent: GranularQLearningAgent,
    env_factory: Callable[[], PortEnvironment],
    episodes: int = 12,
) -> Dict[str, float]:
    metrics: List[Dict[str, float]] = []
    for _ in range(episodes):
        env = env_factory()
        state = env.reset()
        done = False
        while not done:
            state, _, done, _ = env.step(agent.predict(state))
        metrics.append(env.metrics())

    return {
        key: round(float(np.mean([row[key] for row in metrics])), 4)
        for key in metrics[0].keys()
    }


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None


if torch is not None:

    class DQN(nn.Module):
        def __init__(self, state_size: int = 5, action_size: int = 3) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, 96),
                nn.ReLU(),
                nn.Linear(96, 96),
                nn.ReLU(),
                nn.Linear(96, action_size),
            )

        def forward(self, x):
            return self.net(x)


    class DQNAgent:
        """Optional PyTorch DQN agent. Used automatically only when torch is installed."""

        def __init__(
            self,
            state_size: int = 5,
            action_size: int = 3,
            epsilon: float = 1.0,
            epsilon_min: float = 0.05,
            epsilon_decay: float = 0.993,
            gamma: float = 0.95,
            lr: float = 0.001,
        ) -> None:
            self.state_size = state_size
            self.action_size = action_size
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.gamma = gamma
            self.memory = deque(maxlen=6000)
            self.model = DQN(state_size, action_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.loss = nn.SmoothL1Loss()

        def _scale(self, state: State):
            denominator = torch.tensor([60.0, 0.8, 100.0, 10.0, 23.0], dtype=torch.float32)
            return torch.tensor(state, dtype=torch.float32) / denominator

        def choose_action(self, state: State, training: bool = True) -> int:
            if training and random.random() < self.epsilon:
                return random.randrange(self.action_size)
            with torch.no_grad():
                q_values = self.model(self._scale(state))
            return int(torch.argmax(q_values).item())

        def remember(self, state, action, reward, next_state, done) -> None:
            self.memory.append((state, action, reward, next_state, done))

        def replay(self, batch_size: int = 64) -> None:
            if len(self.memory) < batch_size:
                return
            batch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in batch:
                current = self.model(self._scale(state))
                target = current.detach().clone()
                future = 0.0 if done else torch.max(self.model(self._scale(next_state))).item()
                target[action] = reward + self.gamma * future
                loss = self.loss(current, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        def save_model(self, path: Path | str) -> Path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "epsilon": self.epsilon,
                    "gamma": self.gamma,
                },
                path,
            )
            return path

        def load_model(self, path: Path | str) -> None:
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
            self.gamma = checkpoint.get("gamma", self.gamma)
