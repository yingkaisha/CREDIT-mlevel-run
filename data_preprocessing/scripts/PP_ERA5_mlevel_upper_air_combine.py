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

# ------------------------------------------------- #
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
source_dir = conf['zarr_opt']['save_loc_1deg'] + 'upper_air/source/' 
base_dir = conf['zarr_opt']['save_loc_1deg'] + 'upper_air/' 
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

hourly_datetimes = create_hourly_datetime_strings(year)
N_time_gap = len(hourly_datetimes)//6

file_collection = []

for i in range(N_time_gap):
    time_pick = hourly_datetimes[6*i]
    save_name = source_dir + 'ERA5_mlevel_1deg_{}_conserve.nc'.format(time_pick)
    file_collection.append(xr.open_dataset(save_name))

ds_merge_1deg = xr.concat(file_collection, dim='time')

# ========================================================================== #
# chunking
varnames = list(ds_merge_1deg.keys())

for i_var, var in enumerate(varnames):
    ds_merge_1deg[var] = ds_merge_1deg[var].chunk(conf['zarr_opt']['chunk_size_4d_1deg'])

# zarr encodings
dict_encoding = {}

chunk_size_4d = dict(chunks=(conf['zarr_opt']['chunk_size_4d_1deg']['time'],
                             conf['zarr_opt']['chunk_size_4d_1deg']['level'],
                             conf['zarr_opt']['chunk_size_4d_1deg']['latitude'],
                             conf['zarr_opt']['chunk_size_4d_1deg']['longitude']))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    dict_encoding[var] = {'compressor': compress, **chunk_size_4d}

save_name = base_dir + 'ERA5_mlevel_1deg_6h_{}_conserve.zarr'.format(year)
ds_merge_1deg.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)



