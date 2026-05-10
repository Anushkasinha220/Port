from fastapi import FastAPI
from sim.port_env import PortEnvironment, fixed_baseline_policy
import numpy as np

app = FastAPI()

@app.get("/")
def root():
    return {"project": "Port-Optimus", "status": "running"}

@app.get("/predict")
def predict(queue: float, price: float, battery: float, trucks: float, hour: float):
    state = np.array([queue, price, battery, trucks, hour], dtype=np.float32)
    action = int(fixed_baseline_policy(state))
    return {
        "action": action,
        "action_name": ["Rapid Discharge", "Buffered Move", "Idle/Charge"][action],
        "state": {
            "queue": queue,
            "price": price,
            "battery": battery,
            "trucks": trucks,
            "hour": hour
        }
    }

@app.get("/metrics")
def metrics():
    from sim.port_env import PortEnvironment, run_episode
    env = PortEnvironment(seed=42)
    result = run_episode(env, fixed_baseline_policy)
    return result["metrics"]
