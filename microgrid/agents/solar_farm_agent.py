import datetime
from microgrid.environments.solar_farm.solar_farm_env import SolarFarmEnv
import numpy as np


class SolarFarmAgent:
    def __init__(self, env: SolarFarmEnv):
        self.env = env

    def take_decision(self, state, n_fut_time_slots: int = 48):
        return self.take_baseline_decision(state=state, n_fut_time_slots=n_fut_time_slots)

    def take_baseline_decision(self, state, n_fut_time_slots: int = 48):
        # get current State-of-Charge of battery
        current_soc = state['soc']
        # get forecast of PV prod. profile
        pv_profile_forecast = self.env.observation_space["pv_prevision"]
        # get - static - battery parameters
        battery_capa = self.env.battery.capacity
        max_power = self.env.battery.pmax
        charging_efficiency = self.env.battery.efficiency
        # apply very simple policy
        baseline_decision = np.zeros(n_fut_time_slots)
        for t in range(n_fut_time_slots):
            baseline_decision[t] = min(pv_profile_forecast[t],
                                       (battery_capa - current_soc) / charging_efficiency,
                                       max_power)
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
        action = agent.take_decision(state)
        state, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
        print(f"action: {action}, reward: {reward}, cumulative reward: {cumulative_reward}")
        print("State: {}".format(state))
        print("Info: {}".format(action.sum(axis=0)))