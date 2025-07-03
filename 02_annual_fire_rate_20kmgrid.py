# -*- coding: utf-8 -*-
"""
2025
@author: Alba Viana-Soto
Calculate annual forest disturbance rates per grid cell from 1985 to 2023
based on fire patch rasters and a forest land use mask.
"""

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import pandas as pd

# --- Configuration ---
raster_folder = '/.../'
geopackage_path = '/.../grid_20km_southernEU.gpkg'
grid_layer = 'grid_20km_southernEU'
forest_mask_path = '/EFDA/forestlanduse_mask.tif'
output_csv = '/.../annual_disturbancefire_rate_pergrid20km.csv'
pixel_area = 0.09  # hectares per pixel
start_year, end_year = 1985, 2023

# Dynamically generate file paths for all fire rasters from 1985 to 2023
fire_event_rasters = [
    os.path.join(raster_folder, f"{year}_fire_patches.tif")
    for year in range(start_year, end_year + 1)
]

# Load data
print("Loading grid from GeoPackage...")
grid_gdf = gpd.read_file(geopackage_path, layer=grid_layer)
print(f"Loaded {len(grid_gdf)} grid cells.")

print("Loading forest mask...")
with rasterio.open(forest_mask_path) as forest_src:
    forest_mask = forest_src.read(1)
    forest_transform = forest_src.transform
    forest_shape = forest_src.shape

def calculate_disturbance_rate_per_year(fire_event_rasters, grid_gdf, forest_mask, forest_shape, forest_transform):
    n_cells = len(grid_gdf)
    n_years = len(fire_event_rasters)

    total_forest_pixels = np.zeros(n_cells)
    burned_area_per_year = np.zeros((n_cells, n_years))
    disturbance_rate_per_year = np.zeros((n_cells, n_years))

    for index, grid_row in grid_gdf.iterrows():
        print(f"Processing grid cell {index+1}/{n_cells}...")
        grid_geom = grid_row.geometry

        # Create mask for current grid cell
        with rasterio.open(fire_event_rasters[0]) as src:
            grid_mask = geometry_mask(
                [grid_geom], invert=True, out_shape=src.shape, transform=src.transform
            )

        # Calculate forest area within grid cell
        forest_area = np.sum((forest_mask == 1) & grid_mask)
        total_forest_pixels[index] = forest_area

        if forest_area == 0:
            print(f"Grid {index+1}: No forest, skipping.")
            continue

        for year_idx, raster_path in enumerate(fire_event_rasters):
            with rasterio.open(raster_path) as event_src:
                fire_pixels = event_src.read(1) == 2   ## fire agent pixels have value of 2 in EFDA
                fire_masked = fire_pixels & grid_mask & (forest_mask == 1)

                burned_pixels = np.sum(fire_masked)
                burned_area_per_year[index, year_idx] = burned_pixels

                disturbance_rate_per_year[index, year_idx] = (
                    burned_pixels / forest_area
                )

        print(
            f"Grid {index+1}: Forest Area = {forest_area}, "
            f"Burned Area = {burned_area_per_year[index]}, "
            f"Disturbance Rates = {disturbance_rate_per_year[index]}"
        )

    return total_forest_pixels, burned_area_per_year, disturbance_rate_per_year

# Run analysis
print("Calculating disturbance rate per year...")
total_forest_pixels, burned_area_per_year, disturbance_rate_per_year = calculate_disturbance_rate_per_year(
    fire_event_rasters, grid_gdf, forest_mask, forest_shape, forest_transform
)
print("Finished calculating disturbance rate per year.")

# Append results to GeoDataFrame ---
grid_gdf['total_forest_pixels'] = total_forest_pixels

for year_idx, raster_path in enumerate(fire_event_rasters):
    year = int(os.path.basename(raster_path).split('_')[1])
    grid_gdf[f'burned_area_{year}'] = burned_area_per_year[:, year_idx]
    grid_gdf[f'disturbance_rate_{year}'] = disturbance_rate_per_year[:, year_idx]

print("Saving results to CSV...")
grid_gdf.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}.")
