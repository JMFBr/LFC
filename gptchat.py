import matplotlib.pyplot as plt

def simulate_constellation_coverage(num_satellites, swath_width, spatial_resolution):
    # Define the Earth's surface grid (longitude, latitude)
    min_lon, max_lon = -180, 180
    min_lat, max_lat = -90, 90
    lon_step = spatial_resolution
    lat_step = spatial_resolution
    lon_values = range(int(min_lon), int(max_lon), lon_step)
    lat_values = range(int(min_lat), int(max_lat), lat_step)

    # Initialize the coverage matrix
    coverage = [[0] * len(lat_values) for _ in range(len(lon_values))]

    # Simulate coverage for each satellite
    for _ in range(num_satellites):
        # Generate random satellite position
        satellite_lon = random.choice(lon_values)
        satellite_lat = random.choice(lat_values)

        # Calculate coverage bounds
        min_lon_coverage = satellite_lon - swath_width / 2
        max_lon_coverage = satellite_lon + swath_width / 2
        min_lat_coverage = satellite_lat - spatial_resolution / 2
        max_lat_coverage = satellite_lat + spatial_resolution / 2

        # Update coverage matrix
        for i, lon in enumerate(lon_values):
            for j, lat in enumerate(lat_values):
                if min_lon_coverage <= lon <= max_lon_coverage and min_lat_coverage <= lat <= max_lat_coverage:
                    coverage[i][j] += 1

    return coverage, lon_values, lat_values

# Example usage
num_satellites = 5
swath_width = 100  # Assuming in kilometers
spatial_resolution = 0.1  # Assuming in degrees

coverage, lon_values, lat_values = simulate_constellation_coverage(num_satellites, swath_width, spatial_resolution)

# Plot the coverage heatmap
plt.imshow(coverage, extent=[min(lon_values), max(lon_values), min(lat_values), max(lat_values)], origin='lower', cmap='hot')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Earth Observation Constellation Coverage')
plt.colorbar(label='Number of satellites covering each grid point')
plt.show()
