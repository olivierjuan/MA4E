import datetime
from microgrid.environments.data_center.data_center_env import DataCenterEnv
import numpy as np


class DataCenterAgent:
    def __init__(self, env: DataCenterEnv):
        self.env = env
        self.nbr_future_time_slots = env.nb_pdt
        self.Tcom = env.Tcom
        self.Tr = env.Tr
        self.rho = env.rho
        self.COP_HP = env.COP_HP
        self.EER = env.EER
        self.COP_CS = env.COP_CS
        self.max_transfert = env.max_transfert

    def take_decision(self,
                      now: datetime.datetime,               # current datetime
                      manager_signal: np.ndarray,           # in R^nbr_future_time_slots
                      consumption_forecast: np.ndarray,     # in R+^nbr_future_time_slots
                      hotwater_price_forecast: np.ndarray   # in R+^nbr_future_time_slots
                      ) -> np.ndarray:  # in [0,1]^nbr_future_time_slots (heat pump activation profile)
        return np.zeros(self.nbr_future_time_slots)


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    data_center_config = {
        'scenario': 10,
    }
    env = DataCenterEnv(data_center_config, nb_pdt=N)
    agent = DataCenterAgent(env)
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