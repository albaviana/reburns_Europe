# -*- coding: utf-8 -*-
"""
2024
@author: Alba Viana Soto
This script identifies fire events per pixel from the annual fire maps
"""

import os
import numpy as np
import rasterio
from rasterio.enums import Compression
from multiprocessing import Pool

input_folder = "/.../fire_patches/"
output_folder = "/.../fire_patches/number_event/"

class DisturbanceHistory:
    def __init__(self, width, height):
        self.history = np.full((height, width), None, dtype=object)

    def add_disturbance(self, x, y, year):
        if self.history[y, x] is None:
            self.history[y, x] = []
        self.history[y, x].append(year)

def process_chunk(chunk_years, disturbance_history):
    for year in chunk_years:
        raster_path = os.path.join(input_folder, f"{year}_fire_patches.tif") 
        if not os.path.exists(raster_path):
            print(f"File not found: {raster_path}")
            continue
        
        with rasterio.open(raster_path) as src:
            binary_data = src.read(1)
            nodata_value = src.nodata
            for x in range(src.width):
                for y in range(src.height):
                    if binary_data[y, x] == 2 and binary_data[y, x] != nodata_value:   ## fire has a value of 2 in EFDA agent maps
                        disturbance_history.add_disturbance(x, y, year)
            print(f"Adding disturbances for year {year}")

def generate_event_rasters(disturbance_history, width, height):
    for event_num in range(1, 6):
        event_raster = np.zeros((height, width), dtype=np.uint16)
        print(f"Calculating raster for event {event_num}")

        for x in range(width):
            for y in range(height):
                years = disturbance_history.history[y, x]
                if years and event_num <= len(years):
                    sorted_years = sorted(years)
                    event_raster[y, x] = sorted_years[event_num - 1]

        event_raster_path = os.path.join(output_folder, f"{event_num}_fire_event.tif")
        with rasterio.open(
            event_raster_path,
            'w',
            driver='GTiff',
            width=width,
            height=height,
            count=1,
            dtype='uint16',  ## to store values for years from 1985 to 2023
            crs=disturbance_history.crs,
            transform=disturbance_history.transform,
            compress=Compression.lzw  # Set LZW compression
        ) as dst:
            dst.write(event_raster, 1)
            print(f"Wrote raster for event {event_num}")

def main():
    years = range(1985, 2024)
    chunk_size = 50 # Adjust the chunk size as needed
    num_chunks = len(years) // chunk_size + (1 if len(years) % chunk_size != 0 else 0)

    # Process first raster to get the dimensions and CRS
    first_raster_path = os.path.join(input_folder, f"{1985}_fire_patches.tif")
    with rasterio.open(first_raster_path) as src:
        width, height = src.width, src.height
        crs = src.crs
        transform = src.transform
    
    disturbance_history = DisturbanceHistory(width, height)
    disturbance_history.crs = crs
    disturbance_history.transform = transform

    # Process rasters in chunks
    for i in range(num_chunks):
        chunk_years = years[i * chunk_size: (i + 1) * chunk_size]
        process_chunk(chunk_years, disturbance_history)

    # Generate event rasters
    generate_event_rasters(disturbance_history, width, height)

if __name__ == "__main__":
    main()

print("Event rasters tracking disturbance years have been created.")