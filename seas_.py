from glob import glob
# import rioxarray
import xarray as xr
from pandas import to_datetime
import rasterio

# # from cmcrameri import cm as cmc
# # import matplotlib.pyplot as plt
# import cmaps
# cmaps.precip2_17lev
# cmaps.WhiteBlueGreenYellowRed
# cmaps.cmaps_tbr_240_300_r


# define season
SS = 'JF'
SS = 'MAM'
SS = 'JJAS'
SS = 'OND'

# YY = np.r_[2000:2025]
# # YY = np.r_[2024]

dic = {'JF': [1, 2], 'MAM': [3, 4, 5], 'JJAS': [6, 7, 8, 9], 'OND': [10, 11, 12]}
# find months
MM = dic[SS]  # list(map(lambda x:'{:02d}'.format(x), dic[SS]))


idir = 'D:\\SETS\\imergm_07\\'  # path of MONTHLY-IMERG!!

WGS84_WKT = 'GEOGCS["WGS 84",'\
    'DATUM["WGS_1984",'\
        'SPHEROID["WGS 84",6378137,298.257223563,'\
            'AUTHORITY["EPSG","7030"]],'\
        'AUTHORITY["EPSG","6326"]],'\
    'PRIMEM["Greenwich",0,'\
        'AUTHORITY["EPSG","8901"]],'\
    'UNIT["degree",0.0174532925199433,'\
        'AUTHORITY["EPSG","9122"]],'\
    'AUTHORITY["EPSG","4326"]]'


# %% reading & summing up monthly data (IMERG)


flist = [glob(f'{idir}/*{"{:02d}".format(M)}01-S000000*') for M in MM]
# https://stackoverflow.com/a/20112805/5885810  (flattening list of lists)
flist = [item for items in flist for item in items]

imerg = xr.open_mfdataset(
    flist, combine='nested', concat_dim='time', chunks='auto',
    mask_and_scale=True, data_vars=['precipitation'],
    # decode_times=True, decode_cf=True, drop_variables=['mask]'], parallel=True
    )

time = imerg.indexes['time'].to_datetimeindex(time_unit='s')
timo = xr.DataArray(time.days_in_month * 24, dims=['time'], coords=[imerg.time,])

merg = imerg * timo
merg = merg.groupby(group='time.year').sum().compute()
merg = merg.rename_dims({'year': 'time'})
merg = merg.rename_vars({'year': 'time', 'precipitation': 'rain'})
merg['time'] = to_datetime([f'{x}0101' for x in merg.time.data],
                           yearfirst=True, origin='unix')
merg = merg.transpose('time', 'lat', 'lon')

merg.rio.write_crs(rasterio.crs.CRS.from_wkt(WGS84_WKT),
                   grid_mapping_name='spatial_ref', inplace=True,)
merg['rain'].attrs = {'units': 'mm/season'}

# merg.rain.plot(
#     x='lon', y='lat', col='time', col_wrap=6, cmap='cmaps_tbr_240_300_r',
#     # robust=False, vmin=10, vmax=1400, levels=27,  # for JF
#     robust=True,
#     cbar_kwargs={'shrink':0.5, 'aspect':30, 'label':'seasonal rainfall [mm]', 'pad':+.02,}
#     )


# %% XPORT it as NC4 -> WITH.interpretable.CRS! + compression
"""
assigning the "GRID_MAPPING" attribute proved to be of outmost importance
"""

merg.to_netcdf(
    f'./model_input/rainfall_{SS}.nc', mode='w', engine='netcdf4',
    encoding={
        'rain': {'dtype': 'f4', 'zlib': True, 'complevel': 9,
                 'grid_mapping': 'spatial_ref', },
        # 'mask': {'dtype': 'u1','_FillValue': 0, 'grid_mapping': 'spatial_ref'},
        }
    )
