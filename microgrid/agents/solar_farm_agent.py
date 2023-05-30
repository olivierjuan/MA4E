import datetime
from microgrid.environments.solar_farm.solar_farm_env import SolarFarmEnv
from microgrid.agents.internal.check_feasibility import check_solar_farm_feasibility
import numpy as np


class SolarFarmAgent:
    def __init__(self, env: SolarFarmEnv):
        self.env = env
        self.nbr_future_time_slots = env.nb_pdt
        self.delta_t = env.delta_t
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
        baseline_decision = self.take_baseline_decision(soc=soc, pv_forecast=pv_forecast)
        # use format and feasibility "checker"
        check_msg = self.check_decision(load_profile=baseline_decision)
        # format or infeasiblity pb? Look at the check_msg
        if check_msg['format'] != 'ok' or check_msg['infeas'] != 'ok':
            print(f"Format or infeas. errors: {check_msg}")

        return baseline_decision

    def take_baseline_decision(self,
                               soc: float,                  # in [0, battery_capacity]
                               pv_forecast: np.ndarray      # in R+^nbr_future_time_slots
                               ) -> np.ndarray:             # in R^nbr_future_time_slots (battery power profile)
        # get current State-of-Charge of battery
        current_soc = soc
        # get forecast of PV prod. profile
        pv_profile_forecast = pv_forecast
        # apply very simple policy
        baseline_decision = np.zeros(self.nbr_future_time_slots)
        return baseline_decision
        for t in range(self.nbr_future_time_slots):
            baseline_decision[t] = min(
                pv_profile_forecast[t],
                (self.battery_capacity - current_soc) / (self.battery_efficiency * self.delta_t / datetime.timedelta(hours=1)),
                self.battery_pmax
            )
            # update current value of SOC
            current_soc += baseline_decision[t] * self.delta_t / datetime.timedelta(hours=1)
        return baseline_decision

    def check_decision(self, load_profile) -> dict:
        check_msg, check_score = check_solar_farm_feasibility(solar_farm_env=self.env, load_profile=load_profile)
        return check_msg


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
    signal = np.ones(N*2)
    for i in range(N*2):
        state['manager_signal'] = signal
        action = agent.take_decision(**state)
        state, reward, done, info = env.step(action)
        signal
        cumulative_reward += reward
        if done:
            break
        print(f"action: {action}, reward: {reward}, cumulative reward: {cumulative_reward}")
        print("State: {}".format(state))
        print("Info: {}".format(action.sum(axis=0)))