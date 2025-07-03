# -*- coding: utf-8 -*-
"""
2024
@author: Alba Viana Soto
This script extract patches from annual disturbance maps from EFDA v2.1.1. - ESSD 2025
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import label, binary_dilation

def process_rasters(input_dir, output_dir, proximity_threshold=120, chunk_size=20000):   ## tune proximity according to needs and adapt chunk size depending on RAM, threads etc.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for year in range(1985, 2023 + 1):
        input_file = os.path.join(input_dir, f"{year}_disturbance_agent.tif")
        output_file = os.path.join(output_dir, f"{year}_fire_patches.tif")
        
        with rasterio.open(input_file) as src:
            meta = src.meta.copy()
            meta.update(dtype=rasterio.uint32, compress='lzw')
            transform = src.transform
            pixel_size = transform[0]
            dilation_iterations = int(proximity_threshold / pixel_size)
            overlap = 2 * dilation_iterations  # Ensure buffer covers potential patch size across chunks
            
            # Determine chunk sizes
            nrows, ncols = src.height, src.width
            chunk_height = min(chunk_size, nrows)
            chunk_width = min(chunk_size, ncols)
            
            with rasterio.open(output_file, 'w', **meta) as dst:
                for row_start in range(0, nrows, chunk_height):
                    for col_start in range(0, ncols, chunk_width):
                        # Calculate the window position with overlap
                        row_off = max(0, row_start - overlap)
                        col_off = max(0, col_start - overlap)
                        height = min(chunk_height + 2 * overlap, nrows - row_off)
                        width = min(chunk_width + 2 * overlap, ncols - col_off)

                        buffer_window = Window(col_off, row_off, width, height)
                        data = src.read(1, window=buffer_window)
                        
                        # Identify fire pixels
                        fire_pixels = data == 2   ## adapt depending on pixel values (annual mosaics is "1" and 1,2 or 3 for the annual agents mosaics)
                        
                        # Dilate the fire pixels to account for proximity
                        dilated_fire_pixels = binary_dilation(fire_pixels, iterations=dilation_iterations)
                        print(f"calculating pixel proximity for {input_file}")
                        
                        # Label connected components on the dilated array
                        structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
                        labeled_array, num_features = label(dilated_fire_pixels, structure=structure)
                        print(f"*** labelling for {input_file} ***")                                     ## assigns an ID
                        
                        # Extract the relevant window without the overlap
                        result_row_off = overlap if row_start > 0 else 0
                        result_col_off = overlap if col_start > 0 else 0
                        result_window = Window(result_col_off, result_row_off, chunk_width, chunk_height)
                        labeled_result = labeled_array[result_window.toslices()]
                        
                        # Define the writing window without overlap
                        write_window = Window(col_start, row_start, chunk_width, chunk_height)
                        
                        # Write the labeled patches to the output file
                        dst.write(labeled_result.astype(rasterio.uint32), 1, window=write_window)

# Example usage:
input_directory = "/data/EFDA/"
output_directory = "/data/EFDA/fire_patches/"
process_rasters(input_directory, output_directory)
