from typing import Optional, Union, Tuple

import gym
import numpy as np
from gym import spaces
from gym.core import ObsType, ActType
import datetime


class ManagerEnv(gym.Env):
    def __init__(self, agents: dict, manager_config: dict, nb_pdt: int, delta_t: datetime.timedelta = datetime.timedelta(minutes=15), seed: Optional[int] = None):
        self.nb_agents = len(agents)
        self.agents = agents
        self.envs = [agent.env for _, agent in agents.items()]
        self.nb_pdt = nb_pdt
        self.delta_t = delta_t

        self.observation_space = spaces.Dict(
            {
                'datetime': spaces.Space[datetime.datetime]((), np.datetime64, seed),
                'consumptions_prevision': spaces.Box(low=-np.inf, high=np.inf, shape=(self.nb_agents, nb_pdt)),
            }
        )
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(nb_pdt,))
        self.now = None
        self.delta_t = None
        self.n_coord_step = None
        self.microgrid_state = None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        soc, effective_power, penalties = self.battery.charge(action[0], delta_t=self.delta_t)
        self.now += self.delta_t
        return self._step_common(effective_power, penalties)

    def _step_common(self, effective_power, penalties) -> Tuple[ObsType, float, bool, dict]:
        state = {
            'datetime': self.now,
            'manager_signal': np.zeros(self.nb_pdt),
            'soc': self.battery.soc,
            'pv_prevision': self.pv.get_pv_prevision([self.now + i * self.delta_t for i in range(self.nb_pdt)]),
        }
        reward = 0 if penalties == BatteryState.OK else -1e5
        return state, reward, False, {'reward': reward, 'penalties': penalties, 'effective_power': effective_power, 'soc': self.battery.soc, 'datetime': self.now}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None)\
            -> Union[ObsType, Tuple[ObsType, dict]]:
        now = datetime.datetime.now()
        now = now.replace(minute=0, second=0, microsecond=0)
        states = dict()
        actions = dict()
        consumptions = []
        for name, agent in self.agents.items():
            env = agent.env
            state = env.reset(now, self.delta_t, seed=seed, return_info=return_info, options=options)
            action = agent.take_decision(state)
            consumptions.append(env.get_consumption(state, action))
            states[name] = state
            actions[name] = state
        manager_state = {
            'datetime': now,
            'consumptions_prevision': np.array(consumptions)
        }
        return manager_state

    def render(self, mode="human"):
        pass

