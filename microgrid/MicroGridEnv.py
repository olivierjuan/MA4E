import datetime
from typing import Optional, Union, Tuple
import gym
import numpy as np
from gym.core import ObsType, ActType, spaces


class MicroGrid:
    def __init__(self, envs: dict, nb_pdt: int, delta_t: datetime.timedelta = datetime.timedelta(minutes=15), seed: Optional[int] = None):
        self.envs = envs
        self.nb_pdt = nb_pdt
        self.delta_t = delta_t
        self.microgrid_state = None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        responses = {}
        for k, v in self.envs.items():
            responses[k] = v[0].step(action[k])
        self.microgrid_state = {
            'datetime': self.microgrid_state['datetime'] + self.delta_t,
            'consumption': np.sum((v for _, v in action.items()), axis=0)
        }
        state = {'microgrid': self.microgrid_state}
        reward = 0
        done = False
        info = {}
        for k, v in responses.items():
            state[k] = v[0]
            reward += v[1]
            done = done or v[2]
            info[k] = v[3]
        return state, reward, done, info

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None)\
            -> Union[ObsType, Tuple[ObsType, dict]]:
        self.microgrid_state = {
            'datetime': datetime.datetime.now(),
            'consumption': np.zeros(self.nb_pdt)
        }
        state = {'microgrid': self.microgrid_state}
        for k, v in self.envs.items():
            state[k] = v[0].reset(
                self.microgrid_state['datetime'],
                self.delta_t,
                seed=seed,
                return_info=return_info,
                options=options
            )
        return state

    def render(self, mode="human"):
        pass

