import datetime
from microgrid.environments.industrial.industrial_env import IndustrialEnv
import numpy as np


class IndustrialAgent:
    def __init__(self, env: IndustrialEnv):
        self.env = env
        self.nbr_future_time_slots = env.nb_pdt
        self.battery_capacity = env.battery.capacity
        self.battery_pmax = env.battery.pmax
        self.battery_efficiency = env.battery.efficiency

    def take_decision(self,
                      now: datetime.datetime,            # current datetime
                      manager_signal: np.ndarray,        # in R^nbr_future_time_slots
                      soc: float,                        # in [0, battery_capacity]
                      consumption_forecast: np.ndarray   # in R+^nbr_future_time_slots
                      ) -> np.ndarray:                   # in R^nbr_future_time_slots (battery power profile)
        return np.zeros(self.nbr_future_time_slots)


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    industrial_config = {
        'battery': {
            'capacity': 100,
            'efficiency': 0.95,
            'pmax': 25,
        },
        'building': {
            'site': 1,
        }
    }
    env = IndustrialEnv(industrial_config=industrial_config, nb_pdt=N)
    agent = IndustrialAgent(env)
    cumulative_reward = 0
    now = datetime.datetime.now()
    state = env.reset(now, delta_t)
    for i in range(N*2):
        action = agent.take_decision(**state)
        state, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
        print(f"action: {action}, reward: {reward}, cumulative reward: {cumulative_reward}")
        print("State: {}".format(state))