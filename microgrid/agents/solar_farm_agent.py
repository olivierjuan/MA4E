import datetime
from microgrid.environments.solar_farm.solar_farm_env import SolarFarmEnv
import numpy as np


class SolarFarmAgent:
    def __init__(self, env: SolarFarmEnv):
        self.env = env
        self.nbr_future_time_slots = env.nb_pdt
        self.battery_capacity = env.battery.capacity
        self.battery_pmax = env.battery.pmax
        self.battery_efficiency = env.battery.efficiency
        self.pv_pmax = env.pv.pmax

    def take_decision(self,
                      now: datetime.datetime,      # current datetime
                      manager_signal: np.ndarray,  # in R^nbr_future_time_slots
                      soc: float,                  # in [0, battery_capacity]
                      pv_forecast: np.ndarray      # in R+^nbr_future_time_slots
                      ) -> np.ndarray:             # in R^nbr_future_time_slots (battery power profile)
        return self.take_baseline_decision(now=now,
                                           manager_signal=manager_signal,
                                           soc=soc,
                                           pv_forecast=pv_forecast)

    def take_baseline_decision(self,
                               now: datetime.datetime,      # current datetime
                               manager_signal: np.ndarray,  # in R^nbr_future_time_slots
                               soc: float,                  # in [0, battery_capacity]
                               pv_forecast: np.ndarray      # in R+^nbr_future_time_slots
                               ) -> np.ndarray:             # in R^nbr_future_time_slots (battery power profile)
        # get current State-of-Charge of battery
        current_soc = soc
        # get forecast of PV prod. profile
        pv_profile_forecast = pv_forecast
        # apply very simple policy
        baseline_decision = np.zeros(self.nbr_future_time_slots)
        for t in range(self.nbr_future_time_slots):
            baseline_decision[t] = min(
                pv_profile_forecast[t],
                (self.battery_capacity - current_soc) / self.battery_efficiency,
                self.battery_pmax
            )
            # update current value of SOC
            current_soc += baseline_decision[t]
        return baseline_decision


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    solar_farm_config = {
        'battery': {
            'capacity': 100,
            'efficiency': 0.95,
            'pmax': 25,
        },
        'pv': {
            'surface': 100,
            'location': "enpc",  # or (lat, long) in float
            'tilt': 30,  # in degree
            'azimuth': 180,  # in degree from North
            'tracking': None,  # None, 'horizontal', 'dual'
        }
    }
    env = SolarFarmEnv(solar_farm_config=solar_farm_config, nb_pdt=N)
    agent = SolarFarmAgent(env)
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
        print("Info: {}".format(action.sum(axis=0)))