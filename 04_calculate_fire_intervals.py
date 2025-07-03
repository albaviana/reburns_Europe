# -*- coding: utf-8 -*-
"""
2025
@author: Alba Viana-Soto
calculating time in between fire events
"""


import os
import numpy as np
import rasterio

# Parameters
raster_folder = '/data/.../number_events/'
output_folder = '/data/.../intervals/'  # Update with your output folder path
fire_event_rasters = [f'{i}_fire_event.tif' for i in range(1, 6)]  # 1st to 5th fire event rasters

# Function to calculate time between fire events
def calculate_time_between_events(raster_folder, fire_event_rasters, output_folder):
    # Load all the fire event rasters
    fire_event_data = []
    metadata = None

    print(f"Loading fire event rasters...")
    for raster_file in fire_event_rasters:
        file_path = os.path.join(raster_folder, raster_file)
        with rasterio.open(file_path) as src:
            if metadata is None:
                metadata = src.meta.copy()  # Save the metadata of the rasters
            fire_event_data.append(src.read(1))  # Read raster data

    print("Fire event rasters loaded.")
    
    # Loop through consecutive fire events and calculate the difference
    for i in range(1, len(fire_event_data)):
        print(f"Calculating time between fire event {i} and fire event {i+1}...")
        
        # Calculate the difference in years between consecutive events
        current_event = fire_event_data[i]
        previous_event = fire_event_data[i-1]
        
        # Time between events: only where both fire events occurred (non-zero values)
        time_between = np.where((previous_event > 0) & (current_event > 0),
                                current_event - previous_event, 0)
        
        # Clip the values to be within the range of uint8 (0-255)
        time_between_clipped = np.clip(time_between, 0, 255).astype(rasterio.uint8)

        # Update the metadata for the output raster (uint8 and LZW compression)
        metadata.update(dtype=rasterio.uint8, compress='lzw')
        
        # Save the time difference as a new raster
        output_raster = os.path.join(output_folder, f'time_between_{i}_and_{i+1}.tif')
        with rasterio.open(output_raster, 'w', **metadata) as dst:
            dst.write(time_between_clipped, 1)
        
        print(f"Saved time between fire event {i} and {i+1} to {output_raster}")

# Run the function
calculate_time_between_events(raster_folder, fire_event_rasters, output_folder)