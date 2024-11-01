'''
This script collects ERA5 model-level analysis from NCAR/RDA.
6 hourly 1 degree nc files are produced after gathering

Yingkai Sha
ksha@ucar.edu
'''

import os
import sys
import yaml
import dask
import zarr
import numpy as np
import xarray as xr
from glob import glob

import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu
import interp_utils as iu

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

# --------------------------------------------------- #
def create_hourly_datetime_strings(year):
    # Generate hourly date range for the entire year
    date_range = pd.date_range(start=f'{year}-01-01T00', end=f'{year}-12-31T23', freq='H')
    # Format each datetime to the desired string format
    datetime_strings = date_range.strftime('%Y-%m-%dT%H').tolist()
    
    return datetime_strings

# ==================================================================================== #
# get year from input
year = int(args['year'])

config_name = os.path.realpath('../data_config_mlevel_6h.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# save to zarr
base_dir = conf['zarr_opt']['save_loc_1deg'] + 'cloud/source/' 
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

variables_levels = {}
varnames = ['specific_cloud_ice_water_content', 
            'specific_cloud_liquid_water_content',
            'specific_rain_water_content', 
            'specific_snow_water_content']

for varname in varnames:
    variables_levels[varname] = None

ERA5_1h = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1/",
    chunks=None,
    consolidated=True,
    storage_options=dict(token='anon'),)

hourly_datetimes = create_hourly_datetime_strings(year)
N_time_gap = len(hourly_datetimes)//6

count = 0
for i in range(N_time_gap):
    
    time_pick = hourly_datetimes[6*i]
    save_name = base_dir + 'ERA5_mlevel_1deg_cloud_{}_conserve.nc'.format(time_pick)
    
    try:
        ds_test = xr.open_dataset(save_name)
        ds_test.close()
        print('{} Exist'.format(save_name))
    except:
        ERA5_6h = ERA5_1h.sel(time=time_pick)
        ERA5_6h_save = vu.ds_subset_everything(ERA5_6h, variables_levels)
        ERA5_6h_save = ERA5_6h_save.rename({'hybrid': 'level',})
        ds_merge = ERA5_6h_save
        
        # ======================================================================================= #
        # 0.25 deg to 1 deg interpolation using conservative approach
        if count == 0:
            # Define the target 1-degree grid
            lon_1deg = np.arange(0, 360, 1)
            lat_1deg = np.arange(-90, 91, 1)
            target_grid = iu.Grid.from_degrees(lon_1deg, lat_1deg)
            
            lon_025deg = ds_merge['longitude'].values
            lat_025deg = ds_merge['latitude'].values[::-1]
            source_grid = iu.Grid.from_degrees(lon_025deg, lat_025deg)
            
            regridder = iu.ConservativeRegridder(source=source_grid, target=target_grid)
        
        ds_merge = ds_merge.chunk({'longitude': -1, 'latitude': -1})
        ds_merge_1deg = regridder.regrid_dataset(ds_merge)
        
        # Reorder the dimensions for all variables in ds_merge_1deg
        for var in ds_merge_1deg.data_vars:
            # Get the current dimensions of the variable
            current_dims = ds_merge_1deg[var].dims
            
            # If both 'latitude' and 'longitude' are present, reorder them
            if 'latitude' in current_dims and 'longitude' in current_dims:
                # New order: move 'latitude' and 'longitude' to the first two positions, preserve other dimensions
                new_order = [dim for dim in current_dims if dim not in ['latitude', 'longitude']] + ['latitude', 'longitude']
                
                # Transpose the variable to the new order
                ds_merge_1deg[var] = ds_merge_1deg[var].transpose(*new_order)
        
        lon_1deg = np.arange(0, 360, 1)
        lat_1deg = np.arange(-90, 91, 1)
        
        # Add latitude and longitude as coordinates to ds_merge_1deg
        ds_merge_1deg = ds_merge_1deg.assign_coords({
            'latitude': lat_1deg,
            'longitude': lon_1deg
        })
        
        # flip latitude from -90 --> 90 to 90 --> -90
        ds_merge_1deg = ds_merge_1deg.isel(latitude=slice(None, None, -1))
        
        # float64 --> float32
        ds_merge_1deg = ds_merge_1deg.astype(
            {var: np.float32 for var in ds_merge_1deg if ds_merge_1deg[var].dtype == np.float64})
    
        # Convert latitude, longitude, and level coordinates to float32
        ds_merge_1deg = ds_merge_1deg.assign_coords({
            'latitude': ds_merge_1deg['latitude'].astype(np.float32),
            'longitude': ds_merge_1deg['longitude'].astype(np.float32),
            'level': ds_merge_1deg['level'].astype(np.float32)
        })
        
        ds_merge_1deg.to_netcdf(save_name, mode='w', compute=True)
        print('Save to {}'.format(save_name))
        count += 1



