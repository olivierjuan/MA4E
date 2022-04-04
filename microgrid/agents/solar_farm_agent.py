import datetime
from microgrid.environments.solar_farm.solar_farm_env import SolarFarmEnv


class SolarFarmAgent:
    def __init__(self, env: SolarFarmEnv):
        self.env = env

    def take_decision(self, state):
        return self.env.action_space.sample()


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=15)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    battery_config = {
        'capacity': 100,
        'efficiency': 0.95,
        'pmax': 25,
    }
    pv_config = {
        'surface': 100,
        'location': "enpc",  # or (lat, long) in float
        'tilt': 30,  # in degree
        'azimuth': 180,  # in degree from North
        'tracking': None,  # None, 'horizontal', 'dual'
    }
    env = SolarFarmEnv(battery_config=battery_config, pv_config=pv_config, nb_pdt=N)
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