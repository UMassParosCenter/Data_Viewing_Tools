"""
This script queries Paros infrasound sensor data from an InfluxDB database
for a specified time range and sensor, then saves the retrieved data to a local
MAT-file using a generic output path. 

Features:
- Uses pathlib for flexible, cross-platform file paths
- Creates the output directory if it does not exist
- Prompts for secure password input

Intended Use:
- Quickly fetch and store Paros sensor data for offline analysis, preprocessing,
  or plotting.

Dependencies:
- paros_data_grabber (custom module for InfluxDB query and saving)
- pathlib (standard library)

Author: Ethan Gelfand
Date:   2025-08-17
"""

from pathlib import Path
from paros_data_grabber import query_influx_data, save_data

# Define output path using pathlib
output_dir = Path(r"C:\Data\PAROS\Exported_Paros_Data")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "Example_Earthquake.mat"
password = "*****" # input your password here

# Fetch sensor data
data = query_influx_data(
    start_time="2025-05-05T03:03:00",
    end_time="2025-05-05T03:04:00",
    box_id="parost2",
    sensor_id="141929",
    password=password
)

if not data:
    print("No data returned.")
else:
    # Save data to file (convert Path to str if needed)
    save_data(data, str(output_file))
    print(f"Data saved to: {output_file.resolve()}")
