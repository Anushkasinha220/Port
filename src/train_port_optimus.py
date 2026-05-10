from __future__ import annotations

import sys
from pathlib import Path

# Resolve project root robustly whether run as `python -m src.train_port_optimus`
# (GitHub Actions / repo root) or from inside the src/ folder directly.
_THIS_FILE = Path(__file__).resolve()
# If this file is inside a `src` subfolder, go up 2 levels; otherwise go up 1.
PROJECT_ROOT = _THIS_FILE.parents[1] if _THIS_FILE.parent.name == "src" else _THIS_FILE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "port_optimus_q_agent.pkl"

from sim.port_env import PortEnvironment
from src.engine import AgentConfig, GranularQLearningAgent, evaluate_agent
from src.mlops import log_experiment


def main() -> None:
    config = AgentConfig(episodes=320)
    agent = GranularQLearningAgent(config=config, seed=2026)
    env_factory = lambda: PortEnvironment(seed=None)
    agent.train(env_factory, episodes=config.episodes)
    model_path = agent.save_model(MODEL_PATH)
    metrics = evaluate_agent(agent, env_factory, episodes=20)
    run = log_experiment(
        params={
            "agent": "GranularQLearningAgent",
            "episodes": config.episodes,
            "epsilon_min": config.epsilon_min,
            "epsilon_decay": config.epsilon_decay,
            "state": [
                "Queue_Length",
                "Current_Electricity_Price",
                "Port_Battery_Level",
                "Truck_Availability",
                "Hour_of_Day",
            ],
            "actions": {
                "0": "Rapid Discharge",
                "1": "Buffered Move",
                "2": "Idle/Charge",
            },
        },
        metrics=metrics,
        model_path=str(model_path),
    )
    print(f"Port-Optimus run complete: {run['run_id']}")
    print(f"Model saved to: {model_path}")
    print(f"Energy_Efficiency_Ratio: {metrics['Energy_Efficiency_Ratio']}")
    print(f"Throughput_Rate: {metrics['Throughput_Rate']}")


if __name__ == "__main__":
    main()