import datetime
from microgrid.environments.charging_station.charging_station_env import ChargingStationEnv
from microgrid.agents.internal.check_feasibility import check_charging_station_feasibility
import numpy as np


class ChargingStationAgent:
    def __init__(self, env: ChargingStationEnv):
        self.env = env
        self.nbr_evs = env.nb_evs
        self.nbr_future_time_slots = env.nb_pdt
        self.delta_t = env.delta_t
        self.evs_capacity = np.array([ev.battery.capacity for ev in env.evs])
        self.evs_efficiency = np.array([ev.battery.efficiency for ev in env.evs])
        self.evs_pmax = np.array([ev.battery.pmax for ev in env.evs])
        self.station_pmax = env.pmax_site

    def take_decision(self,
                      now: datetime.datetime,             # current datetime
                      manager_signal: np.ndarray,         # in R^nbr_future_time_slots
                      soc: np.ndarray,                    # in [0, battery_capacity(i)]^nbr_evs
                      is_plugged_prevision: np.ndarray   # in {0, 1}^(nb_evs, nbr_future_time_slots)
                      ) -> np.ndarray:  # in [0, evs_pmax(i)]^(nb_evs, nbr_future_time_slots)
                                        # (each charger power profile)
        baseline_decision = self.take_baseline_decision(soc=soc, is_plugged_prevision=is_plugged_prevision)
        # use format and feasibility "checker"
        check_msg = self.check_decision(load_profile=baseline_decision, is_plugged_forecast=is_plugged_prevision)
        # format or infeasiblity pb? Look at the check_msg
        if check_msg['format'] != 'ok' or check_msg['infeas'] != 'ok':
            print(f"Format or infeas. errors: {check_msg}")

        return baseline_decision

    def take_baseline_decision(self,
                               soc: np.ndarray,                   # in [0, battery_capacity(i)]^nbr_evs
                               is_plugged_prevision: np.ndarray   # in {0, 1}^(nb_evs, nbr_future_time_slots)
                               ) -> np.ndarray:
        # get current State-of-Charge of EV batterieS
        current_soc = soc.copy()
        # apply very simple policy
        baseline_decision = np.zeros([self.nbr_evs, self.nbr_future_time_slots])
        delta_t_float = self.delta_t / datetime.timedelta(hours=1)
        for t in range(self.nbr_future_time_slots):
            # get number of plugged EVs at this time-slot
            n_ev_plugged = sum(is_plugged_prevision[:, t])
            # uniform share
            p_ev = self.station_pmax / n_ev_plugged if n_ev_plugged > 0 else 0
            # ensure compatibility with battery constraints
            for i_ev in range(self.nbr_evs):
                if is_plugged_prevision[i_ev, t] == 1:
                    delta_soc_max = (self.evs_capacity[i_ev] - current_soc[i_ev])
                    max_power_to_soc_max = delta_soc_max / (delta_t_float * self.evs_efficiency[i_ev])
                    baseline_decision[i_ev, t] = \
                        min(p_ev,
                            max_power_to_soc_max,
                            self.evs_pmax[i_ev])
                current_soc[i_ev] += baseline_decision[i_ev, t] * delta_t_float * self.evs_efficiency[i_ev]
                # update current value of SOC
        return baseline_decision

    def check_decision(self, load_profile, is_plugged_forecast: np.ndarray) -> dict:
        check_msg, check_score, cs_dep_soc_penalty = \
            check_charging_station_feasibility(charging_station_env=self.env, load_profile=load_profile,
                                               is_plugged_forecast=is_plugged_forecast, dep_soc_penalty=5)
        return check_msg


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
    now = datetime.datetime.now().replace(hour=0)
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