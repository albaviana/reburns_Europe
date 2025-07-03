# -*- coding: utf-8 -*-
"""
2025
@author: Alba Viana-Soto
Calculate reburn fractions per year since 2005
"""


import os
import traceback
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
from multiprocessing import Pool

# --- Parameters ---
start_year = 1985
end_year = 2023
reburn_window = 20
burn_value = 2
batch_size = 100
num_processes = 20
output_path = '/.../results/per_year_20yr_interval_reburns_proportions_grid20km.csv'

raster_folder = '/data/'
grid_shapefile = '/data/.../grid_20km_southernEU.gpkg'

# Load grid 
grids = gpd.read_file(grid_shapefile, layer='grid_20km_southernEU')

# Read raster for a year and bounds
def read_yearly_raster(year, bounds):
    file_path = os.path.join(raster_folder, f'{year}_fire_patches.tif')
    if not os.path.exists(file_path):
        print(f" Missing raster for year {year}: {file_path}")
        return None, None, None

    with rasterio.open(file_path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        raster_data = src.read(1, window=window, masked=True)
        transform = src.window_transform(window)
        shape = raster_data.shape

        if isinstance(raster_data, np.ma.MaskedArray):
            raster_data = raster_data.filled(0 if np.issubdtype(raster_data.dtype, np.integer) else np.nan)

        return raster_data, transform, shape

# Core computation: per grid cell
def calculate_reburn_proportion(grid_geom, grid_id):
    bounds = grid_geom.bounds
    yearly_reburns = {}
    yearly_total_burns = {}

    for year in range(2005, end_year + 1):
        try:
            current_raster, transform, shape = read_yearly_raster(year, bounds)
            if current_raster is None:
                continue

            mask = geometry_mask([grid_geom], invert=True, transform=transform, out_shape=shape)
            burned_current = (np.round(current_raster) == burn_value) & mask
            total_burn = (current_raster > 0) & mask
            total_burn_count = np.sum(total_burn)

            reburn_count = 0
            for prev_year in range(max(start_year, year - reburn_window), year):
                prev_raster, _, _ = read_yearly_raster(prev_year, bounds)
                if prev_raster is None:
                    continue
                reburn_count += np.sum(burned_current & (np.round(prev_raster) == burn_value))

            yearly_reburns[year] = reburn_count
            yearly_total_burns[year] = total_burn_count

        except Exception as e:
            print(f"Error in Grid {grid_id}, Year {year}: {e}")
            traceback.print_exc()

    return yearly_reburns, yearly_total_burns

# Worker: Process a single grid cell
def process_grid_cell(grid, grid_id):
    print(f"▶️ Processing Grid {grid_id}")
    try:
        reburns, totals = calculate_reburn_proportion(grid['geometry'], grid_id)
        results = [
            (grid_id, year, reburns[year], totals[year],
             reburns[year] / totals[year] if totals[year] > 0 else 0)
            for year in reburns
        ]
        return results
    except Exception as e:
        print(f"Error in grid {grid_id}: {e}")
        traceback.print_exc()
        return []

# --- Parallel processor ---
def parallel_process_grid_cells(grids, processes):
    with Pool(processes=processes) as pool:
        results = pool.starmap(
            process_grid_cell,
            [(grid, grid['id']) for _, grid in grids.iterrows()]
        )
    return [r for result in results for r in result]  # flatten list

# --- Batch processor ---
def batch_process_grid_cells(grids):
    all_results = []
    num_batches = (len(grids) - 1) // batch_size + 1

    for batch_idx in range(num_batches):
        print(f"Batch {batch_idx + 1}/{num_batches}")
        start, end = batch_idx * batch_size, (batch_idx + 1) * batch_size
        batch = grids.iloc[start:end]

        try:
            batch_results = parallel_process_grid_cells(batch, processes=num_processes)
            all_results.extend(batch_results)

            # Save interim results
            pd.DataFrame(all_results, columns=['Grid_ID', 'Year', 'Reburn_Count', 'Total_Burned_Area', 'Reburn_Proportion']).to_csv(output_path, index=False)
        except Exception as e:
            print(f"Error in batch {batch_idx + 1}: {e}")
            traceback.print_exc()

    return all_results

# --- Run analysis ---
results = batch_process_grid_cells(grids)
results_df = pd.DataFrame(results, columns=['Grid_ID', 'Year', 'Reburn_Count', 'Total_Burned_Area', 'Reburn_Proportion'])
results_df.to_csv(output_path, index=False)

print("Reburn analysis completed.")
print(results_df.head())

