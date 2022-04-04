import datetime
from microgrid.environments.industrial.industrial_env import IndustrialEnv


class IndustrialAgent:
    def __init__(self, env: IndustrialEnv):
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
    building_config = {
        'site': 1,
    }
    env = IndustrialEnv(battery_config=battery_config, building_config=building_config, nb_pdt=N)
    agent = IndustrialAgent(env)
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