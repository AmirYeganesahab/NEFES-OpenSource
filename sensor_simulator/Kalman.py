from filterpy.kalman import KalmanFilter
import numpy as np


from temp import ForestFireTemperatureSimulator
import numpy as np
import matplotlib.pyplot as plt

# Example of using the advanced simulator with a forest fire
latitude = 30  # Specify the latitude of the point on Earth
target_average_temperature = 25  # Specify the target average temperature
min_temperature = 20  # Specify the minimum temperature
max_temperature = 60  # Specify the maximum temperature
oscillation_factor = 5  # Specify the oscillation factor
outlier_probability = 0  # Specify the probability of having an outlier
outlier_magnitude = 0  # Specify the magnitude of the outliers
fire_start_day = 10  # Specify the day when the forest fire starts (during summer)
fire_duration_days = 3  # Specify the duration of the forest fire
fire_magnitude = 50  # Specify the magnitude of the temperature spike during the fire
fire_smoothness = 3  # Specify the smoothness of the fire spike

forest_fire_simulator = ForestFireTemperatureSimulator(
    latitude,
    target_average_temperature,
    min_temperature,
    max_temperature,
    oscillation_factor,
    outlier_probability,
    outlier_magnitude,
    fire_start_day,
    fire_duration_days,
    fire_magnitude,
    fire_smoothness,
)

# Simulate temperature for 265 days
num_days_to_simulate = 366
output = forest_fire_simulator.simulate_temperature_for_days(num_days_to_simulate)
# Plotting
temps_vs_hours_of_doy = output["temps_vs_hours_of_doy"]

fig, ax = plt.subplots(figsize=(12, 6))
cax = ax.imshow(
    temps_vs_hours_of_doy,
    cmap="coolwarm",
    aspect="auto",
    extent=[0, 244, 0, num_days_to_simulate],
)
ax.set_title(f"Hourly Temperature Variation, Latitude {latitude}°")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Day")
cbar = fig.colorbar(cax, label="Temperature (°C)")
plt.show()
# Plotting
temps, mjds = output["temperature_time_series"].items()
outliers = output["outliers"]
nmjds = np.array(mjds[1])[outliers]
ntmps = np.array(temps[1])[outliers]
# plt.plot(mjds[1],temps[1])
fig, ax = plt.subplots(figsize=(24, 6))
cax = ax.plot(mjds[1], temps[1])
ax.scatter(nmjds, ntmps, color="r")
# ax.set_title(f'Hourly Temperature Variation, Latitude {latitude}°')
# ax.set_xlabel('Hour of Day')
# ax.set_ylabel('Day')
# cbar = fig.colorbar(cax, label='Temperature (°C)')
plt.show()

time_diff = mjds[1][1] - mjds[1][0]
initial_rate_of_change = (temps[1][1] - temps[1][0]) / time_diff


# Create a Kalman filter with 2 state variables (temperature and its rate of change)
kf = KalmanFilter(dim_x=2, dim_z=1)

# Define state transition matrix (assuming constant velocity model)
kf.F = np.array([[1, time_diff], [0, 1]])

# Define measurement matrix (assuming we directly measure temperature)
kf.H = np.array([[1, 0]])

# Set initial state and covariance
kf.x = np.array([temps[1][0], initial_rate_of_change])
kf.P *= np.eye(2)  # Initial covariance matrix

# Process and measurement noise
kf.Q *= 0.01  # Process noise covariance
kf.R *= 0.1  # Measurement noise covariance

# Simulate measurements (replace this with your actual temperature sensor data)
measurements = temps[1]
# Apply Kalman filter to estimate temperature
filtered_temperatures = []

plt.ion()
for measurement in measurements:
    # Predict step
    kf.predict()

    # Update step
    kf.update(measurement)
    # Save the filtered temperature estimate
    filtered_temperatures.append(kf.x[0])
    # Plot the current measurement and prediction
    plt.scatter(
        len(filtered_temperatures),
        measurement,
        color="red",
        marker="x",
        label="Measurement",
    )
    plt.plot(
        range(1, len(filtered_temperatures) + 1),
        filtered_temperatures,
        color="blue",
        label="Filtered Temperature",
    )
    plt.xlabel("Time Step")
    plt.ylabel("Temperature")
    plt.legend()
    plt.title("Kalman Filter Prediction and Measurements")
    plt.draw()
    plt.pause(0.1)
    plt.clf()

# Plot the results

# Plot the final results
plt.plot(mjds[1], measurements, label="True Temperature")
plt.plot(mjds[1], filtered_temperatures, label="Filtered Temperature")
plt.xlabel("Time Step")
plt.ylabel("Temperature")
plt.legend()
plt.title("Kalman Filter Final Results")
plt.show()
