from __future__ import annotations

import re

import numpy as np

from api import metrics, predict, root
from sim.port_env import PortEnvironment, fixed_baseline_policy, run_episode
from src.engine import AgentConfig, GranularQLearningAgent, evaluate_agent
from src.mlops import next_run_id


def test_environment_reset_returns_five_feature_state():
    env = PortEnvironment(seed=7)
    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (5,)


def test_environment_step_returns_metrics_history():
    env = PortEnvironment(seed=7)
    state = env.reset()
    next_state, reward, done, info = env.step(1)
    assert next_state.shape == (5,)
    assert isinstance(reward, float)
    assert done is False
    assert info["action_name"] == "Buffered Move"
    assert len(env.history) == 1


def test_fixed_baseline_policy_returns_valid_action():
    env = PortEnvironment(seed=11)
    action = fixed_baseline_policy(env.reset())
    assert action in {0, 1, 2}


def test_baseline_episode_produces_core_metrics():
    result = run_episode(PortEnvironment(seed=42), fixed_baseline_policy)
    metric_names = {
        "Energy_Efficiency_Ratio",
        "Throughput_Rate",
        "Total_Processed",
        "Total_Energy",
        "Total_Cost",
        "Average_Wait_Penalty",
        "Average_Safety",
    }
    assert metric_names.issubset(result["metrics"])


def test_q_learning_smoke_training_and_evaluation():
    config = AgentConfig(episodes=2, epsilon_decay=0.9)
    agent = GranularQLearningAgent(config=config, seed=2026)
    agent.train(lambda: PortEnvironment(seed=None), episodes=2)
    evaluation = evaluate_agent(agent, lambda: PortEnvironment(seed=12), episodes=2)
    assert len(agent.training_rewards) == 2
    assert evaluation["Throughput_Rate"] >= 0


def test_mlops_run_id_format():
    run_id = next_run_id()
    assert re.match(r"^ALPHA-\d{4}-\d{3}$", run_id)


def test_fastapi_root_and_predict_contracts():
    assert root()["status"] == "running"
    response = predict(queue=12, price=0.31, battery=50, trucks=5, hour=14)
    assert response["action"] in {0, 1, 2}
    assert "action_name" in response


def test_fastapi_metrics_contract():
    response = metrics()
    assert response["Total_Processed"] >= 0
    assert 0 <= response["Average_Safety"] <= 100
