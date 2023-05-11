from typing import Optional, Union, Tuple

import gym
import numpy as np
from gym import spaces
from gym.core import ObsType, ActType

from microgrid.assets.battery import Battery, BatteryState
import datetime

from microgrid.assets.pv import PV


class SolarFarmEnv(gym.Env):
    def __init__(self, solar_farm_config: dict, nb_pdt=24, seed: Optional[int] = None):
        self.battery_config = solar_farm_config.get('battery', {})
        self.pv_config = solar_farm_config.get('pv', {})
        self.nb_pdt = nb_pdt
        self.battery = Battery(**self.battery_config)
        self.pv = PV(**self.pv_config)

        self.observation_space = spaces.Dict(
            {
                'now': spaces.Space[datetime.datetime]((), np.datetime64, seed),
                'manager_signal': spaces.Box(low=-np.inf, high=np.inf, shape=(nb_pdt,)),
                'soc': spaces.Box(low=0.0, high=self.battery.capacity, shape=(1,)),
                'pv_forecast': spaces.Box(low=0.0, high=self.pv.pmax, shape=(nb_pdt,)),
            }
        )
        self.action_space = spaces.Box(low=self.battery.pmin, high=self.battery.pmax, shape=(nb_pdt,))
        self.now = None
        self.delta_t = None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        soc, effective_power, penalties = self.battery.charge(action[0], delta_t=self.delta_t)
        self.now += self.delta_t
        effective_action = action[:]
        effective_action[0] = effective_power
        return self._step_common(effective_action, penalties)

    def try_step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        effective_power, penalties = self.battery.check_power(action[0], delta_t=self.delta_t)
        effective_action = action[:]
        effective_action[0] = effective_power
        return self._step_common(effective_action, penalties)

    def _step_common(self, effective_action, penalties) -> Tuple[ObsType, float, bool, dict]:
        state = {
            'now': self.now,
            'manager_signal': np.zeros(self.nb_pdt),
            'soc': self.battery.soc,
            'pv_forecast': self.pv.get_pv_prevision([self.now + i * self.delta_t for i in range(self.nb_pdt)]),
        }
        reward = 0 if penalties == BatteryState.OK else -1e5
        return state, reward, False, {'reward': reward, 'penalties': penalties, 'effective_action': effective_action, 'soc': self.battery.soc, 'now': self.now}

    def reset(self, *args, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None)\
            -> Union[ObsType, Tuple[ObsType, dict]]:
        self.now, self.delta_t = tuple(args)
        self.battery.reset()
        state = {
            'now': self.now,
            'manager_signal': np.zeros(self.nb_pdt),
            'soc': self.battery.soc,
            'pv_forecast': self.pv.get_pv_prevision([self.now + i * self.delta_t for i in range(self.nb_pdt)]),
        }
        return state

    def get_consumption(self, state: ObsType,  action: ActType) -> np.ndarray:
        return -state['pv_forecast'] + action

    def render(self, mode="human"):
        pass

