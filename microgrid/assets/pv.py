from collections import defaultdict

import numpy as np
import datetime
from typing import Union, Tuple
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sunpos(when, location, refraction):
    # Extract the passed data
    year, month, day, hour, minute, second = when.utctimetuple()[:6]
    # timezone = when.utcoffset() / datetime.timedelta(hours=1)
    timezone = 0
    latitude, longitude = location
    # Math typing shortcuts
    rad, deg = np.radians, np.degrees
    sin, cos, tan = np.sin, np.cos, np.tan
    asin, atan2 = np.arcsin, np.arctan2
    # Convert latitude and longitude to radians
    rlat = rad(latitude)
    rlon = rad(longitude)
    # Decimal hour of the day at Greenwich
    greenwichtime = hour - timezone + minute / 60 + second / 3600
    # Days from J2000, accurate from 1901 to 2099
    daynum = (
        367 * year
        - 7 * (year + (month + 9) // 12) // 4
        + 275 * month // 9
        + day
        - 730531.5
        + greenwichtime / 24
    )
    # Mean longitude of the sun
    mean_long = daynum * 0.01720279239 + 4.894967873
    # Mean anomaly of the Sun
    mean_anom = daynum * 0.01720197034 + 6.240040768
    # Ecliptic longitude of the sun
    eclip_long = (
        mean_long
        + 0.03342305518 * sin(mean_anom)
        + 0.0003490658504 * sin(2 * mean_anom)
    )
    # Obliquity of the ecliptic
    obliquity = 0.4090877234 - 0.000000006981317008 * daynum
    # Right ascension of the sun
    rasc = atan2(cos(obliquity) * sin(eclip_long), cos(eclip_long))
    # Declination of the sun
    decl = asin(sin(obliquity) * sin(eclip_long))
    # Local sidereal time
    sidereal = 4.894961213 + 6.300388099 * daynum + rlon
    # Hour angle of the sun
    hour_ang = sidereal - rasc
    # Local elevation of the sun
    elevation = asin(sin(decl) * sin(rlat) + cos(decl) * cos(rlat) * cos(hour_ang))
    # Local azimuth of the sun
    azimuth = atan2(
        -cos(decl) * cos(rlat) * sin(hour_ang),
        sin(decl) - sin(rlat) * sin(elevation),
    )
    # Convert azimuth and elevation to degrees
    azimuth = into_range(deg(azimuth), 0, 360)
    elevation = into_range(deg(elevation), -180, 180)
    # Refraction correction (optional)
    if refraction:
        targ = rad((elevation + (10.3 / (elevation + 5.11))))
        elevation += (1.02 / tan(targ)) / 60
    # Return azimuth and elevation in degrees
    return round(azimuth, 2), round(elevation, 2)


def into_range(x, range_min, range_max):
    shiftedx = x - range_min
    delta = range_max - range_min
    return (((shiftedx % delta) + delta) % delta) + range_min


class PV:
    _known_locations = {
        "enpc": (48.84115450291472, 2.587556234801854),
    }

    def __init__(
            self,
            surface=1,
            efficiency=0.15,
            location: Union[str, Tuple[float, float]] = "enpc",
            tilt=30,
            azimuth=180,
            tracking: Union[None, str] = None
            ):
        self.surface = surface
        self.efficiency = efficiency
        self.pmax = self.surface * self.efficiency
        self.tilt = tilt
        self.azimuth = azimuth
        self.tracking = tracking
        if isinstance(location, str):
            if location.lower() not in PV._known_locations:
                print(f"Unknown location {location}, falling back to enpc")
            self.location = PV._known_locations.get(location.lower(), PV._known_locations["enpc"])
        else:
            self.location = location

    def get_power(self, when: datetime.datetime):
        azimuth, altitude = sunpos(when, self.location, True)

        solar_zenith = np.radians(90 - altitude)
        solar_azimuth = np.radians(azimuth)
        am = 1 / np.cos(solar_zenith)

        def f_projection(varazimuth, vartilt):
            surface_tilt = np.radians(vartilt)
            surface_azimuth = np.radians(varazimuth)

            projection = (
                    np.cos(surface_tilt) * np.cos(solar_zenith) +
                    np.sin(surface_tilt) * np.sin(solar_zenith) *
                    np.cos(solar_azimuth - surface_azimuth)
            )
            #dni = 1.353 * math.pow(0.7, math.pow(am, 0.678)) * projection
            return projection

        pv_azimuth = self.azimuth
        pv_tilt = self.tilt
        if self.tracking is not None:
            if self.tracking == "horizontal":
                pv_azimuth = float(opt.minimize(lambda x: -f_projection(x, pv_tilt), azimuth, bounds=[(90., 270.)]).x)
                self.azimuth = pv_azimuth
            elif self.tracking == "dual":
                pv_azimuth, pv_tilt = tuple(
                    opt.minimize(lambda x: -f_projection(*list(x)), np.array([azimuth, pv_tilt]),
                                 bounds=[(90., 270.), (20, 45)]).x)
                self.azimuth = pv_azimuth
                self.tilt = pv_tilt
            elif self.tracking == "vertical":
                pv_tilt = float(opt.minimize(lambda x: -f_projection(pv_azimuth, x), pv_tilt, bounds=[(20, 45)]).x)
                self.tilt = pv_tilt

        projection = f_projection(pv_azimuth, pv_tilt)

        # GH 1185
        projection = np.clip(projection, -1, 1)
        if projection <= 0 or am <= 0:
            return 0
        aoi = np.rad2deg(np.arccos(projection))

        dni = 1.353 * np.power(0.7, np.power(am, 0.678))
        irradiance = dni * projection  # np.cos(np.radians(aoi))

        power = irradiance * self.surface * self.efficiency
        return power

    def get_pv_prevision(self, datetimes: [datetime.datetime]):
        return np.array(list(map(self.get_power, datetimes)))


if __name__ == "__main__":
    dates = []
    values = defaultdict(list)
    for type in [None, 'horizontal', 'dual', 'vertical']:
        dates = []
        a = PV(tracking=type)
        for t in range(24*10):
            when = [2019, 7, 1, 0, 0, 0, 1]
            when[4] = 5 * t % 60
            when[3] = t // 10
            pos = sunpos(when, a.location, True)
            power = a.get_power(when)
            dates.append(datetime(*when))
            values[type].append(power)
    v = np.array(list(values.values()))
    serie = pd.DataFrame(np.array(list(values.values())).T, index=dates)
    print(serie)
    serie.plot()
    plt.show()

    dates = []
    values = defaultdict(list)
    for tilt in range(10, 90, 10):
        dates = []
        a = PV(tilt=tilt, tracking='dual')
        for t in range(24*10):
            when = [2019, 7, 1, 0, 0, 0, 1]
            when[4] = 5 * t % 60
            when[3] = t // 10
            pos = sunpos(when, a.location, True)
            power = a.get_power(when)
            dates.append(datetime(*when))
            values[tilt].append(power)
    v = list(values.values())
    serie = pd.DataFrame(np.array(list(values.values())).T, index=dates)
    print(serie)
    serie.plot()
    plt.show()