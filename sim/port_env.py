from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


Action = int
State = np.ndarray


@dataclass
class PortStepInfo:
    hour: int
    action_name: str
    units_processed: int
    energy_used: float
    energy_cost: float
    congestion: float
    queue_length: int
    battery_level: float
    electricity_price: float
    truck_availability: int
    wait_penalty: float
    safety_score: float


class PortEnvironment:
    """Small custom RL environment for autonomous port logistics.

    State:
        [queue_length, electricity_price, battery_level, truck_availability, hour]

    Actions:
        0: Rapid Discharge - process fast, consume battery, pay higher energy cost.
        1: Buffered Move - process slower with better energy efficiency.
        2: Idle/Charge - wait and charge battery when possible.
    """

    ACTIONS = {
        0: "Rapid Discharge",
        1: "Buffered Move",
        2: "Idle / Charge",
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        episode_hours: int = 24,
        base_ship_arrival_rate: float = 2.4,
        battery_capacity: float = 100.0,
        initial_battery: float = 72.0,
        max_trucks: int = 9,
        storm: bool = False,
        peak_season: bool = False,
    ) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.episode_hours = episode_hours
        self.base_ship_arrival_rate = base_ship_arrival_rate
        self.battery_capacity = battery_capacity
        self.initial_battery = initial_battery
        self.max_trucks = max_trucks
        self.storm = storm
        self.peak_season = peak_season
        self.reset()

    def reset(self) -> State:
        self.hour = 0
        self.queue_length = int(self.np_rng.integers(7, 15))
        self.battery_level = float(self.initial_battery)
        self.truck_availability = int(self.np_rng.integers(4, self.max_trucks + 1))
        self.total_processed = 0
        self.total_energy = 0.0
        self.total_cost = 0.0
        self.total_wait = 0.0
        self.history: List[Dict[str, float]] = []
        return self._state()

    def _arrival_rate(self) -> float:
        multiplier = 2.0 if self.peak_season else 1.0
        morning_wave = 0.7 if 6 <= self.hour <= 10 else 0.0
        night_wave = 0.4 if 20 <= self.hour <= 23 else 0.0
        return (self.base_ship_arrival_rate + morning_wave + night_wave) * multiplier

    def _electricity_price(self) -> float:
        peak = 0.42 if 17 <= self.hour <= 21 else 0.0
        shoulder = 0.16 if 8 <= self.hour <= 16 else 0.0
        night_discount = -0.08 if self.hour <= 5 else 0.0
        noise = self.rng.uniform(-0.025, 0.025)
        return round(max(0.11, 0.22 + peak + shoulder + night_discount + noise), 3)

    def _solar_input(self) -> float:
        if 6 <= self.hour <= 18:
            daylight = math.sin(((self.hour - 6) / 12.0) * math.pi)
            storm_factor = 0.35 if self.storm else 1.0
            return max(0.0, 9.5 * daylight * storm_factor)
        return 0.0

    def _congestion(self) -> float:
        queue_pressure = min(1.0, self.queue_length / 42.0)
        truck_pressure = 1.0 - (self.truck_availability / max(1, self.max_trucks))
        return round(min(1.0, 0.72 * queue_pressure + 0.28 * truck_pressure), 3)

    def _state(self) -> State:
        return np.array(
            [
                float(self.queue_length),
                float(self._electricity_price()),
                float(self.battery_level),
                float(self.truck_availability),
                float(self.hour),
            ],
            dtype=np.float32,
        )

    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, float]]:
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action {action}. Expected one of {list(self.ACTIONS)}")

        price = self._electricity_price()
        congestion = self._congestion()
        available_work = min(self.queue_length, self.truck_availability)
        processed = 0
        energy_used = 0.0
        wait_penalty = 0.0

        if action == 0:
            processed = min(available_work, 7)
            energy_used = processed * (4.4 + congestion * 1.7)
            if self.battery_level < energy_used:
                shortfall = energy_used - self.battery_level
                processed = max(0, processed - int(math.ceil(shortfall / 5.2)))
                energy_used = min(self.battery_level, processed * (4.4 + congestion * 1.7))
            wait_penalty = self.queue_length * congestion * 0.62
            cost_penalty = 1.45
        elif action == 1:
            processed = min(available_work, 4)
            energy_used = processed * (2.35 + congestion * 0.65)
            wait_penalty = self.queue_length * congestion * 0.42
            cost_penalty = 0.92
        else:
            processed = 0
            energy_used = 0.0
            wait_penalty = self.queue_length * (0.72 + congestion)
            cost_penalty = 0.15

        charge = self._solar_input()
        if action == 2:
            low_price_bonus = 7.0 if price <= 0.22 else 2.5
            charge += low_price_bonus

        self.battery_level = max(
            0.0,
            min(self.battery_capacity, self.battery_level - energy_used + charge),
        )
        self.queue_length -= processed

        arrivals = int(self.np_rng.poisson(self._arrival_rate()))
        self.queue_length = min(60, self.queue_length + arrivals)
        self.truck_availability = int(self.np_rng.integers(3, self.max_trucks + 1))

        energy_cost = energy_used * price * cost_penalty
        reward = (processed * 10.0) - (energy_cost * 2.6) - wait_penalty
        safety_score = max(0.0, 100.0 - congestion * 45.0 - max(0, self.queue_length - 35) * 1.2)

        self.total_processed += processed
        self.total_energy += energy_used
        self.total_cost += energy_cost
        self.total_wait += wait_penalty

        row = {
            "hour": self.hour,
            "action": action,
            "action_name": self.ACTIONS[action],
            "units_processed": processed,
            "energy_used": round(energy_used, 3),
            "energy_cost": round(energy_cost, 3),
            "congestion": congestion,
            "queue_length": self.queue_length,
            "battery_level": round(self.battery_level, 3),
            "electricity_price": price,
            "truck_availability": self.truck_availability,
            "wait_penalty": round(wait_penalty, 3),
            "safety_score": round(safety_score, 3),
            "arrivals": arrivals,
            "solar_input": round(charge, 3),
        }
        self.history.append(row)

        self.hour += 1
        done = self.hour >= self.episode_hours
        return self._state(), float(reward), done, row

    def metrics(self) -> Dict[str, float]:
        throughput_rate = self.total_processed / max(1, self.episode_hours)
        energy_efficiency = self.total_processed / max(1.0, self.total_energy)
        avg_wait = self.total_wait / max(1, len(self.history))
        avg_safety = float(np.mean([h["safety_score"] for h in self.history])) if self.history else 100.0
        return {
            "Energy_Efficiency_Ratio": round(energy_efficiency, 4),
            "Throughput_Rate": round(throughput_rate, 4),
            "Total_Processed": round(float(self.total_processed), 4),
            "Total_Energy": round(self.total_energy, 4),
            "Total_Cost": round(self.total_cost, 4),
            "Average_Wait_Penalty": round(avg_wait, 4),
            "Average_Safety": round(avg_safety, 4),
        }

    def render_lanes(self) -> Dict[str, str]:
        queue_icons = " ".join(["📦"] * min(12, self.queue_length))
        truck_icons = " ".join(["🚚"] * min(8, self.truck_availability))
        ship_icons = " ".join(["🚢"] * max(1, min(5, math.ceil(self.queue_length / 12))))
        return {
            "ships": ship_icons,
            "containers": queue_icons,
            "trucks": truck_icons,
        }


def fixed_baseline_policy(state: State) -> Action:
    queue, price, battery, trucks, hour = state
    if battery < 22:
        return 2
    if queue >= 12 and trucks >= 4:
        return 0
    return 1


def run_episode(env: PortEnvironment, policy, max_steps: Optional[int] = None) -> Dict[str, object]:
    state = env.reset()
    done = False
    steps = 0
    while not done:
        action = int(policy(state))
        state, _, done, _ = env.step(action)
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return {"history": env.history, "metrics": env.metrics()}
