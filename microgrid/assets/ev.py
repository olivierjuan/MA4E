import datetime
import os
import pandas as pd
import numpy as np
from random import randint

from microgrid.assets.battery import Battery


class EV:
    @staticmethod
    def random():
        return EV(ev=randint(1, 10), day=randint(1, 365))

    def __init__(self, ev=1, day=1, capacity=50, pmax=7, pmin=0):
        self.ev = ev
        self.day = day
        self.battery = Battery(capacity=capacity, pmax=pmax, pmin=pmin)
        self.data = pd.read_csv(os.path.join(os.path.dirname(__file__), f'scenarios/ev/data.csv'), delimiter=';')

    def reset(self):
        self.battery.reset()
        self.day = randint(1, 365)

    def get_is_plugged(self, when: datetime.datetime, start: datetime.datetime):
        delta_day = (when - datetime.datetime.fromordinal(start.date().toordinal())) // datetime.timedelta(days=1)
        delta_hour = (when - datetime.datetime.fromordinal(start.date().toordinal())) // datetime.timedelta(hours=1)
        day = (self.day + delta_day - 1) % 365 + 1
        date = datetime.date(year=2014, month=1, day=1) + datetime.timedelta(days=day)
        dep, arr = tuple(self.data.query(f'day == "{date:%d/%m/%Y}" and ev_id == {self.ev}')[['time_slot_dep', 'time_slot_arr']].values[0])
        delta_hour = delta_hour % 24
        if dep <= delta_hour < arr:
            return 0
        return 1

    def get_is_plugged_prevision(self, datetimes: [datetime.datetime]):
        res = []
        if len(datetimes) > 0:
            start = datetimes[0]
            res = np.array(list(map(lambda x: self.get_is_plugged(x, start), datetimes)))
            pdt = (start - datetime.datetime.fromordinal(start.date().toordinal())) // datetime.timedelta(days=15)
            if pdt == 24*4-1:
                self.day = self.day % 365 + 1
        return res

    def check_power(self, power, delta_t=datetime.timedelta(minutes=15)):
        return self.battery.check_power(power, delta_t)

    def charge(self, power, delta_t=datetime.timedelta(minutes=15)):
        return self.battery.charge(power, delta_t)

    def get_soc(self, when: datetime.datetime):
        if self.get_is_plugged(when, when):
            return self.battery.soc
        return 0


if __name__ == '__main__':
    b = EV(ev=1)
    print(b.data)
    print(b.get_is_plugged(datetime.datetime.now(), datetime.datetime.now()))
    print(b.get_is_plugged_prevision([datetime.datetime.now(), datetime.datetime.now()+datetime.timedelta(minutes=15)]))
