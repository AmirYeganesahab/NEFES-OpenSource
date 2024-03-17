import numpy as np
from astropy.time import Time
from datetime import datetime, timedelta


class ForestFireTemperatureSimulator:
    """simulates temperature sensor"""

    def __init__(
        self,
        latitude=0,
        target_average_temperature=25,
        min_temperature=23,
        max_temperature=120,
        oscillation_factor=5,
        outlier_probability=0.05,
        outlier_magnitude=3,
        fire_start_day=150,
        fire_duration_days=11,
        fire_magnitude=15,
        fire_smoothness=3,
        start_date=(2023, 1, 1, 0),
    ):
        self.latitude = latitude
        self.target_average_temperature = target_average_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.oscillation_factor = oscillation_factor
        self.outlier_probability = outlier_probability
        self.outlier_magnitude = outlier_magnitude
        self.fire_start_day = fire_start_day
        self.fire_duration_days = fire_duration_days
        self.fire_magnitude = fire_magnitude
        self.fire_smoothness = fire_smoothness
        self.start_date = start_date

    def calculate_solar_declination(self, day_of_year):
        """Calculate solar declination angle"""
        return 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)

    def calculate_hourly_insolation(self, solar_declination, hour_of_day):
        """Calculate hourly insolation based on latitude and solar declination"""
        day_length = (
            2
            * np.arccos(-np.tan(np.radians(self.latitude)) * np.tan(solar_declination))
            / np.pi
            * 24
        )
        hour_angle = 15 * (hour_of_day - 12)
        if -day_length / 2 < hour_angle < day_length / 2:
            insolation = (
                1361
                * np.cos(np.radians(self.latitude))
                * np.cos(np.radians(solar_declination))
                * np.cos(np.radians(hour_angle))
            )
        else:
            insolation = 0
        return insolation

    def add_outlier(self, temperature):
        """Introduce occasional outliers with smaller deviations"""
        if np.random.rand() < self.outlier_probability:
            return (
                temperature
                + np.random.uniform(-self.outlier_magnitude, self.outlier_magnitude),
                True,
            )
        else:
            return temperature, False

    def calculate_seasonal_variation(self, day_of_year):
        """Simulate seasonal temperature variation using a sinusoidal function"""
        return 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    def calculate_fire_effect(self, day):
        """Simulate the effect of a forest fire on temperature with a smoother spike"""
        if self.fire_start_day <= day < self.fire_start_day + self.fire_duration_days:
            fire_days = np.arange(0, self.fire_duration_days)
            fire_spike = self.fire_magnitude * np.exp(
                -((fire_days - self.fire_smoothness / 2) ** 2)
                / (2 * (self.fire_smoothness / 2) ** 2)
            )
            return fire_spike[day - self.fire_start_day]
        return 0

    def calculate_temperature(self, day_of_year, hour_of_day):
        """Simulate temperature using the Earth Energy
        Balance model with a target average temperature"""
        solar_declination = np.radians(self.calculate_solar_declination(day_of_year))
        insolation = self.calculate_hourly_insolation(solar_declination, hour_of_day)

        # Adjust the temperature calculation to force
        # the time series to have a target average temperature
        temperature = (insolation / (4 * 5.67e-8)) ** 0.25 - 273.15
        temperature += self.target_average_temperature - np.mean(temperature)
        temperature += self.calculate_seasonal_variation(day_of_year)
        temperature += self.oscillation_factor * np.sin(2 * np.pi * hour_of_day / 24)
        temperature += self.calculate_fire_effect(day_of_year)
        temperature = np.clip(temperature, self.min_temperature, self.max_temperature)

        # Add occasional outliers
        temperature = self.add_outlier(temperature)

        return temperature

    def simulate_temperature_for_days(self, num_days, every_x_hour=6):
        """simulated temperature signal of sensor for 'num_days' days and every 'every_x_hours'"""
        start_doy = self.calculate_day_of_year(*self.start_date[:3])
        # Simulate temperature for the specified number of days
        hours_of_day = list(np.arange(0, 24, every_x_hour))
        temperatures_by_day = []
        time_series = []
        mean_of_days = []
        mjds = []
        doys = []
        outliers = []
        for doy in range(start_doy, start_doy + num_days + 1):
            temperatures = [
                self.calculate_temperature(doy, hour) for hour in hours_of_day
            ]
            tempse = [tmp for tmp, _ in temperatures]
            outlier = [bl for _, bl in temperatures]
            mjd = [
                self.calculate_modified_julian_date(self.start_date[0], doy, hour)
                for hour in hours_of_day
            ]
            temperatures_by_day.append(tempse)
            time_series.extend(tempse)
            mean_of_days.append(np.mean(tempse))
            doys.append(doy)
            mjds.extend(mjd)
            outliers.extend(outlier)

        return {
            "temps_vs_hours_of_doy": np.array(temperatures_by_day),
            "temperature_time_series": {"temps": time_series, "mjds": mjds},
            "mean_temps_of_days": {"temps": np.array(mean_of_days), "doys": doys},
            "outliers": outliers,
        }

    def calculate_modified_julian_date(self, year, doy, hour):
        """Create an Astropy Time object with the specified inputs"""
        t = Time(
            datetime(year, 1, 1) + timedelta(days=doy - 1, hours=int(hour)), scale="utc"
        )
        # Get the Modified Julian Date (MJD)
        mjd = t.mjd
        return mjd

    def convert_mjd_to_datetime(self, mjd):
        """Create an Astropy Time object from the MJD"""
        t = Time(mjd, format="mjd", scale="utc")

        # Extract year, month, day, and hour
        year = t.datetime.year
        month = t.datetime.month
        day = t.datetime.day
        hour = t.datetime.hour + t.datetime.minute / 60.0 + t.datetime.second / 3600.0

        return year, month, day, hour

    def calculate_day_of_year(self, year, month, day):
        """get day of the year from year,month and day"""
        days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Check for leap year
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            days_in_month[2] = 29

        doy = sum(days_in_month[:month]) + day
        return doy
