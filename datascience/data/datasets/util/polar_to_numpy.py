import os
import pandas as pd
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt


class Polar:
    """
    From an 'Adrena compatible' .pol file, construct a polar in which an interpolation function is given.
    The latter allow users to approximate the speed of the boat for any TWA and Wind speed.

    # TODO Take into account sailect as well. Might use swell, currents and so forth.
    # TODO Improve the fact that the interpolation is not persistent. Might be a limitation in computational time and consistency as well.
    """
    def __init__(self, path_polar_file):
        self.polar_raw = pd.read_csv(path_polar_file, sep="\t", index_col=0)

        # first dim = TWA, second = Windspeed
        self.values = self.polar_raw.values
        # TWA
        self.twa = self.polar_raw.index.values
        # windspeed
        self.windspeed = self.polar_raw.columns.values.astype(int)

    def speed_boat(self, twa, wind_speed):
        """
        Provide an estimation of the boat speed given the TWA and Windspeed conditions.
        Currently it assumes the boat has symmetric performances.

        If out of the bounds, will set to speed=0knts.

        :param twa: true wind angle (towards angle of wind)
        :param wind_speed: true wind speed
        :return: An estimation of the boat speed in such conditions.
        """

        twa = twa if twa < 180 else 360 - twa  # current polar is only defined on [0,180] degrees
        try:
            speed = interpolate.interpn(points=(self.twa, self.windspeed),
                                        values=self.values, xi=(twa, wind_speed), method="linear")[0]
            # should be better to work on vector instead of one by one
        except ValueError:
            speed = 0

        return speed

    def print1D(self, wind_speed):
        """"
        :param wind_speed: Wind speed for which interpolation is plotted.
        :return: a plot of (actual + modelled) speeds in a 2D plot (TWA, Boatspeed).
        """
        x = np.arange(0, 180.1, 10)
        y_hat = [self.speed_boat(twa=twa, wind_speed=wind_speed) for twa in x] # should be better to work on vector instead

        plt.plot(x, y_hat, '-', label="model")
        plt.plot(self.twa, self.polar_raw[str(wind_speed)], '.', label="Given polar")
        plt.xlabel("TWA")
        plt.ylabel("Boat Speed (knts)")
        plt.legend()
        plt.title("Interpolation of the Polar (for WindSpeed: " + str(wind_speed) + "knts)")

        plt.show()

    def print2D(self):
        """"
        #TODO Print (actual + modelled) speeds in a 3D plot (TWA, windspeed, Boatspeed).
        """
        pass


if __name__ == "__main__":

    pol = Polar(path_polar_file=os.path.join(os.path.dirname(__file__), "../rawdata/polar", "class40.pol"))

    boat_speed = pol.speed_boat(twa=10, wind_speed=2)
    print(boat_speed)

    pol.print1D(wind_speed=12)
