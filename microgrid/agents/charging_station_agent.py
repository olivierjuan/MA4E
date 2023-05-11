import datetime
from microgrid.environments.charging_station.charging_station_env import ChargingStationEnv
import numpy as np


class ChargingStationAgent:
    def __init__(self, env: ChargingStationEnv):
        self.env = env
        self.nbr_evs = env.nb_evs
        self.nbr_future_time_slots = env.nb_pdt
        self.evs_capacity = np.array([ev.battery.capacity for ev in env.evs])
        self.evs_pmax = np.array([ev.battery.pmax for ev in env.evs])
        self.station_pmax = env.pmax_site

    def take_decision(self,
                      now: datetime.datetime,             # current datetime
                      manager_signal: np.ndarray,         # in R^nbr_future_time_slots
                      soc: np.ndarray,                    # in [0, battery_capacity(i)]^nbr_evs
                      is_plugged_prevision: np.ndarray   # in {0, 1}^(nb_evs, nbr_future_time_slots)
                      ) -> np.ndarray:  # in [0, evs_pmax(i)]^(nb_evs, nbr_future_time_slots)
                                        # (each charger power profile)
        return np.zeros((self.nbr_evs, self.nbr_future_time_slots))

    def take_baseline_decision(self):
        return 1


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    evs_config = [
        {
            'capacity': 40,
            'pmax': 22,
        },
        {
            'capacity': 40,
            'pmax': 22,
        },
        {
            'capacity': 40,
            'pmax': 3,
        },
        {
            'capacity': 40,
            'pmax': 3,
        },
    ]
    station_config = {
        'pmax': 40,
        'evs': evs_config
    }
    env = ChargingStationEnv(station_config=station_config, nb_pdt=N)
    agent = ChargingStationAgent(env)
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