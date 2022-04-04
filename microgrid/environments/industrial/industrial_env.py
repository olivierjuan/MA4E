from typing import Optional, Union, Tuple

import gym
import numpy as np
from gym import spaces
from gym.core import ObsType, ActType

from microgrid.assets.battery import Battery, BatteryState
import datetime

from microgrid.assets.building import Building


class IndustrialEnv(gym.Env):
    def __init__(self, battery_config: dict, building_config: dict, nb_pdt=24, seed: Optional[int] = None):
        self.battery_config = battery_config
        self.building_config = building_config
        self.nb_pdt = nb_pdt
        self.battery = Battery(**battery_config)
        self.building = Building(**building_config)

        self.observation_space = spaces.Dict(
            {
                'datetime': spaces.Space[datetime.datetime]((), np.datetime64, seed),
                'manager_signal': spaces.Box(low=-np.inf, high=np.inf, shape=(nb_pdt,)),
                'soc': spaces.Box(low=0.0, high=self.battery.capacity, shape=(1,)),
                'consumption_prevision': spaces.Box(low=0.0, high=np.inf, shape=(nb_pdt,)),
            }
        )
        self.action_space = spaces.Box(low=-self.battery.pmax, high=self.battery.pmax, shape=(nb_pdt,))
        self.now = None
        self.delta_t = None
        self.n_coord_step = None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        soc, effective_power, penalties = self.battery.charge(action[0], delta_t=self.delta_t)
        self.now += self.delta_t
        return self._step_common(effective_power, penalties)

    def try_step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        effective_power, penalties = self.battery.check_power(action[0], delta_t=self.delta_t)
        return self._step_common(effective_power, penalties)

    def _step_common(self, effective_power, penalties) -> Tuple[ObsType, float, bool, dict]:
        state = {
            'datetime': self.now,
            'manager_signal': np.zeros(self.nb_pdt),
            'soc': self.battery.soc,
            'consumption_prevision': self.building.get_conso_prevision([self.now + i * self.delta_t for i in range(self.nb_pdt)]),
        }
        reward = 0 if penalties == BatteryState.OK else -1e5
        return state, reward, False, {'reward': reward, 'penalties': penalties, 'effective_action': effective_power, 'soc': self.battery.soc, 'datetime': self.now}

    def reset(self, *args, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None)\
            -> Union[ObsType, Tuple[ObsType, dict]]:
        self.now, self.delta_t = tuple(args)
        self.building.reset()
        state = {
            'datetime': self.now,
            'manager_signal': np.zeros(self.nb_pdt),
            'soc': self.battery.soc,
            'consumption_prevision': self.building.get_conso_prevision([self.now + i * self.delta_t for i in range(self.nb_pdt)]),
        }
        return state

    def get_consumption(self, state: ObsType,  action: ActType) -> np.ndarray:
        return state['consumption_prevision'] + action

    def render(self, mode="human"):
        pass

