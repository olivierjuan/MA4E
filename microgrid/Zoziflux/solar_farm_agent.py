import datetime
from microgrid.environments.solar_farm.solar_farm_env import SolarFarmEnv
from microgrid.agents.internal.check_feasibility import check_solar_farm_feasibility
import numpy as np
import matplotlib.pyplot as plt

l = 20

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
        baseline_decision, B = self.take_baseline_decision(soc=soc, pv_forecast=pv_forecast, manager_signal=manager_signal)
        # use format and feasibility "checker"
        check_msg = self.check_decision(load_profile=baseline_decision)
        # format or infeasiblity pb? Look at the check_msg
        print(f"Format or infeas. errors: {check_msg}")

        return baseline_decision, B

    def take_baseline_decision(self,
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
        #fonction de bénéfice
        B = 0
        manager_signal2 = [20*np.sin(i/20)+40 for i in range(len(manager_signal))]
        for t in range(self.nbr_future_time_slots):
            if manager_signal2[t] < l : # charge
                baseline_decision[t] = min(
                    pv_profile_forecast[t],
                    (self.battery_capacity - current_soc) / (self.battery_efficiency * self.delta_t / datetime.timedelta(hours=1)),
                    self.battery_pmax
                )
                # update current value of SOC
                current_soc += baseline_decision[t] * self.delta_t / datetime.timedelta(hours=1) * self.battery_efficiency
            else : # décharge = vente
                baseline_decision[t] = - min(
                    self.battery_pmax,
                    current_soc * self.battery_efficiency / (self.delta_t / datetime.timedelta(hours=1)) + 0.00001
                )
                # update profit
                B += - manager_signal2[t] * baseline_decision[t]
                # update current value of SOC
                current_soc += baseline_decision[t] * self.delta_t / datetime.timedelta(hours=1) / self.battery_efficiency + pv_profile_forecast[t]
                if current_soc < 0:
                    print('error')

        #update profit at the end
        B += manager_signal2[len(manager_signal)-1] * current_soc * self.battery_efficiency / (self.delta_t / datetime.timedelta(hours=1))
        return baseline_decision, B


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
    B_hist = []
    for i in range(N*2):
        action, B = agent.take_decision(**state)
        B_hist.append(B)
        state, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
        print(f"action: {action}, reward: {reward}, cumulative reward: {cumulative_reward}")
        print("State: {}".format(state))
        print("Info: {}".format(action.sum(axis=0)))
        print(B_hist)
        print(np.sum(B_hist))
    X = [i for i in range(len(B_hist))]
    plt.scatter(X,B_hist)
    plt.show()