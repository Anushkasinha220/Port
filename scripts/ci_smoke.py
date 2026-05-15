from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.port_env import PortEnvironment, fixed_baseline_policy, run_episode
from src.engine import AgentConfig, GranularQLearningAgent, evaluate_agent


OUT = ROOT / "experiments" / "ci_smoke_metrics.json"


def main() -> None:
    agent = GranularQLearningAgent(AgentConfig(episodes=4), seed=2026)
    agent.train(lambda: PortEnvironment(seed=None), episodes=4)
    rl_metrics = evaluate_agent(agent, lambda: PortEnvironment(seed=42), episodes=3)
    baseline_metrics = run_episode(PortEnvironment(seed=42), fixed_baseline_policy)["metrics"]

    payload = {
        "status": "passed",
        "training_episodes": len(agent.training_rewards),
        "rl_metrics": rl_metrics,
        "baseline_metrics": baseline_metrics,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Port-Optimus smoke training: PASSED")
    print(f"training_episodes={payload['training_episodes']}")
    print(f"rl_throughput={rl_metrics['Throughput_Rate']}")
    print(f"rl_energy_efficiency={rl_metrics['Energy_Efficiency_Ratio']}")
    print(f"baseline_throughput={baseline_metrics['Throughput_Rate']}")
    print(f"artifact={OUT}")


if __name__ == "__main__":
    main()
