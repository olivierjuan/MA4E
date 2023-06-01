import datetime
from microgrid.environments.data_center.data_center_env import DataCenterEnv
from microgrid.agents.internal.check_feasibility import check_data_center_feasibility
import numpy as np
from scipy.stats import bernoulli


def bernoulli_param(delta_price: float, delta_price_max: float, delta_price_min: float):
    """
    Calculate the parameter of a Bernouilli law, based on price difference btw Hot-Water and elec. versus min. and max.
    differences observed over all time-slots
    Args:
        delta_price: current price difference
        delta_price_max: max difference observed on the horizon considered
        delta_price_min: min difference on same horizon

    Returns:
        Bernoulli param, in [0, 1]
    """
    if delta_price_max == delta_price_min:
        return 0.5
    else:
        return (delta_price - delta_price_min) / (delta_price_max - delta_price_min)


class DataCenterAgent:
    def __init__(self, env: DataCenterEnv):
        self.env = env
        self.nbr_future_time_slots = env.nb_pdt
        self.delta_t = env.delta_t
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
        baseline_decision = self.take_baseline_decision(manager_signal=manager_signal,
                                                        hotwater_price_forecast=hotwater_price_forecast)
        # use format and feasibility "checker"
        check_msg = self.check_decision(load_profile=baseline_decision, consumption_forecast=consumption_forecast)
        # format or infeasiblity pb? Look at the check_msg
        if check_msg['format'] != 'ok' or check_msg['infeas'] != 'ok':
            print(f"Format or infeas. errors: {check_msg}")

        return baseline_decision

    def take_baseline_decision(self,
                               manager_signal: np.ndarray,  # in R^nbr_future_time_slots
                               hotwater_price_forecast: np.ndarray   # in R+^nbr_future_time_slots
                               ) -> np.ndarray:
        delta_price_max = max(hotwater_price_forecast - manager_signal)
        delta_price_min = min(hotwater_price_forecast - manager_signal)
        # Random decision, based on respective elec. and Hot-Water prices
        baseline_decision = np.zeros(self.nbr_future_time_slots)
        if delta_price_max > 0:  # otherwise not profitable to use elec. to produce Hot-Water
            for t in range(self.nbr_future_time_slots):
                # set Bernoulli param based on diff. between hot-water and elec. (MG) price
                random_choice = \
                    bernoulli.rvs(p=bernoulli_param(delta_price=hotwater_price_forecast[t] - manager_signal[t],
                                                    delta_price_max=delta_price_max, delta_price_min=delta_price_min),
                                  size=1)
                if random_choice[0] == 1:
                    baseline_decision[t] = self.max_transfert / (self.COP_HP * self.delta_t.total_seconds() / 3600)

        return baseline_decision

    def check_decision(self, load_profile, consumption_forecast):
        check_msg, check_score = check_data_center_feasibility(data_center_env=self.env, load_profile=load_profile,
                                                               it_load_profile=consumption_forecast)
        return check_msg


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