import warnings

# # https://stackoverflow.com/a/9134842/5885810     (supress warning by message)
# warnings.filterwarnings('ignore', message='You will likely lose important projection '\
#                         'information when converting to a PROJ string from another format')
# # WOS doesn't deal with "ecCodes"
# warnings.filterwarnings('ignore', message='Failed to load cfgrib - most likely '\
#                         'there is a problem accessing the ecCodes library.')
# # because the "EPSG_CODE = 42106" is not a standard proj?
# warnings.filterwarnings('ignore', message="GeoDataFrame's CRS is not "\
#                         "representable in URN OGC format")
# # https://github.com/slundberg/shap/issues/2909    (suppresing the one from libpysal 4.7.0)
# warnings.filterwarnings('ignore', message=".*`Geometry` class will deprecated '\
#                         'and removed in a future version of libpysal*")
# # # https://github.com/slundberg/shap/issues/2909    (suppresing the one from numba 0.59.0)
# # warnings.filterwarnings('ignore', message=".*The 'nopython' keyword.*")

# https://stackoverflow.com/a/248066/5885810
from os.path import abspath, basename, dirname, join
parent_d = dirname(__file__)    # otherwise, will append the path.of.the.tests
# parent_d = './'               # to be used in IPython

from gc import collect
import numpy as np
import pandas as pd
# # https://stackoverflow.com/a/65562060/5885810  (ecCodes in WOS)
import xarray as xr
import rioxarray as rio
# from pyproj import CRS
import pyproj as pp
import netCDF4 as nc4
import geopandas as gpd
from scipy import stats
from scipy.ndimage import gaussian_filter, uniform_filter
from numpy import random as npr
from statsmodels.distributions.copula.api import GaussianCopula

# import dask.array as da
from dask import array, delayed

from osgeo import gdal
# https://gdal.org/api/python_gotchas.html#gotchas-that-are-by-design-or-per-history
# https://github.com/OSGeo/gdal/blob/master/NEWS.md#ogr-370---overview-of-changes
if gdal.__version__.__getitem__(0) == '3':# enable exceptions for GDAL<=4.0
    gdal.UseExceptions()
    # gdal.DontUseExceptions()
    # gdal.__version__ # wos_ '3.9.2' # linux_ '3.7.0'

from rasterio import fill
# from rasterio import crs as rcrs
from zoneinfo import ZoneInfo
from datetime import timedelta, timezone, datetime
from dateutil.tz import tzlocal
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

import libpysal as ps
from pointpats import PoissonPointProcess, random, Window  # , PointPattern

import matplotlib.pyplot as plt
from functools import reduce
from operator import iconcat, itemgetter
from parameters import *
from pdfs_ import betas, circular, elevation, field, masking, forecasting


# np.set_printoptions(threshold = np.inf)
np.set_printoptions(linewidth = 140)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
STORM [STOchastic Rainfall Model] produces realistic regional or watershed rainfall under various
climate scenarios based on empirical-stochastic selection of historical rainfall characteristics.

# Based on Manuel F. Rios Gaona et al.
# [ https://doi.org/10.5194/gmd-17-5387-2024 ]

version name: STORM3

Authors:
    Manuel F. Rios Gaona 2024
Date created : 2023/05/11
Last modified: 2024/10/11
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
02-09-2024:
previously in the conda-environment "fitter" (from pip) was installed.
"fitter" last version (1.7.1) downgraded "numpy" 2.1.0 to 1.26.4.
take-home message:
    installing "pointpats" (2.5.0) adds "numpy" 2.1.0 to the environment (washing out?? numpy's
    pip-installation) which generated NO PROBLEMS when testing FITTER (with np's conda-forge)!!
"numpy" 2.1.0 had to be downgraded to 2.0.2 to avoid problems with "numba" 0.60;
and didn't cause problems when running "fitter" either.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# %% constants/switches

ptot_or_kmean = 1  # 1 if seasonal.rain sampled; 0 if taken from shp.kmeans
capmax_or_not = 0  # 1 if using MAXD_RAIN as capping limit; 0 if using iMAX
output_stats_ = 0  # 1 if willing to produce CSV.file; 0 saves some ram.mem
tunnin = 11


minmax_radius = max([X_RES, Y_RES]) / 1e3  # in km (function of resolution)
SEED_YEAR = SEED_YEAR if SEED_YEAR else datetime.now().year  # SEED_YEAR = []?

# convert DATE_ORIGIN into 'datetime'
# https://stackoverflow.com/a/623312/5885810
# https://stackoverflow.com/q/70460247/5885810  (timezone no pytz)
# https://stackoverflow.com/a/65319240/5885810  (replace timezone)
date_origen = datetime.strptime(DATE_ORIGIN, '%Y-%m-%d').replace(
    tzinfo=ZoneInfo(TIME_ZONE))


def update_par(args, **kwargs):
    func = kwargs.get('fun', None)
    # https://stackoverflow.com/a/2083375/5885810  (exec global... weird)
    if isinstance(args, dict):
        for x in list(args.keys()):
            exec(f'globals()["{x}"] = {func}')
    else:
        for x in list(vars(args).keys()):
            exec(f'globals()["{x}"] = args.{x}')


def replicate_():
    # replicate scalars for NUMSIMS (if only one was passed)
    scalar = {
        'PTOT_SC': PTOT_SC, 'PTOT_SF': PTOT_SF,
        'STORMINESS_SC': STORMINESS_SC, 'STORMINESS_SF': STORMINESS_SF,
        }
    n_scal = np.array(list(map(len, list(scalar.values()))))
    maxnum  = max(NUMSIMS, NUMSIMYRS)
    if n_scal.all() and np.unique(n_scal) < maxnum:
        update_par(scalar, fun=f'np.repeat(args[x], {maxnum})')
    # pick the counter to call 'the scalars'
    n_sim_y = ['nsim', 'simy'][np.argmax((NUMSIMS, NUMSIMYRS))]
    # print(PTOT_SC, PTOT_SF)
    return n_sim_y


n_sim_y = replicate_()


# %% nc-file creation

def nc_bytes():
    """
    scales 'down' floats to integers.\n
    Input: none.\n
    Output (sets up the following globals) ->
    SCL : float; multiplicative scaling factor.
    ADD : float; additive scaling factor.
    MINIMUM : int; minimum integer allowed.
    iMAX: float; maximum allowed float (for rainfall).\n
    """
    global SCL, ADD, MINIMUM, iMAX

    # 'u' for UNSIGNED.INT  ||  'i' for SIGNED.INT  ||  'f' for FLOAT
    # number of Bytes (1, 2, 4 or 8) to store the RAINFALL variable (into)
    INTEGER = int(RAINFMT[-1])
    MINIMUM = 1  # 1 because i need 0 for filling
    MAXIMUM = +(2**(INTEGER * 8)) - 1  # 65535 (largest unsigned 16bits-int; 0 also counts)
    # MAXIMUM = +(2**(4 * 8)) - 1  # 4294967296 (largest unsigned 32bits-int)
    # # if RAINFMT=='i2' (signed 16bit-int)
    # MINIMUM = -(2**(INTEGER * 8 - 1))  # -32768 (smallest signed 16bits-int)
    # MAXIMUM = +(2**(INTEGER * 8 - 1) -1)  # +32767 (largest signed 16bits-int)

    # NORMALIZING THE RAINFALL SO IT CAN BE STORED AS 16-BIT INTEGER
    # https://stackoverflow.com/a/59193141/5885810  (scaling 'integers')
    # https://stats.stackexchange.com/a/70808/354951  (normalize data 0-1)
    iMIN = 0.

# # run your own (customized) tests
# temp = 3.14159
# epsilon = [0.006, 0.005, 0.004, 0.003, 0.002, 0.001, .01]
# for epsn in epsilon:
#     seed = 1500
#     while temp > epsn:
#         temp = (seed - iMIN) / (MAXIMUM - MINIMUM - 1)
#         seed -= 1
#     print(f'starting in {iMIN}, you\'d need a max. of '\
#           f'~{seed + 1} to guarantee an epsilon of {epsn}')
# # starting in 0.0, you'd need a max. of ~ 393 to guarantee an epsilon of 0.006
# # starting in 0.0, you'd need a max. of ~ 327 to guarantee an epsilon of 0.005
# # starting in 0.0, you'd need a max. of ~ 262 to guarantee an epsilon of 0.004
# # starting in 0.0, you'd need a max. of ~ 196 to guarantee an epsilon of 0.003
# # starting in 0.0, you'd need a max. of ~ 131 to guarantee an epsilon of 0.002
# # starting in 0.0, you'd need a max. of ~  65 to guarantee an epsilon of 0.001
# # starting in 0.0, you'd need a max. of ~1501 to guarantee an epsilon of 0.01
# # starting in 0.0, you'd need a max. of: 429496 to guarantee an epsilon of 0.0001  # (for INTEGER==4)

    # if you want a larger precision (or your variable is in the 'low' scale,
    # ...say Summer Temperatures in Celsius) you must/could lower this limit.
    iMAX = PRECISION * (MAXIMUM - MINIMUM - 1) + iMIN
    # 131.066 -> for MINIMUM==1
    # 131.068 -> for MINIMUM==0
    SCL = (iMAX - iMIN) / (MAXIMUM - MINIMUM - 1) # -1 because 0 doesn't count
    ADD = iMIN - SCL * MINIMUM

    # # testing
    # allv = PRECISION * (np.linspace(MINIMUM, MAXIMUM - 1, MAXIMUM - 1) - 1) + iMIN
    # # allv = np.arange(iMIN, iMAX + PRECISION, PRECISION)
    # allv.shape  # (65534,)
    # allv[-1] == iMAX  # np.True_
    # allt = ((allv - ADD) / SCL).round(0).astype(RAINFMT)
    # # am i missing some value because of rounding??
    # np.flatnonzero(np.diff(allt) != 1)  # array([], dtype=int64)
    # vall = ((allt * SCL) + ADD).round(4)
    # voll = base_round((allt * SCL) + ADD, method='nearest', base=PRECISION, round=4)
    # (vall == voll).all()  # np.True_

    # if storing as FLOAT
    if RAINFMT[0] == 'f':
        SCL = 1.
        ADD = 0.
        MINIMUM = 0.


def nc_file_iv(nc, **kwargs):
    """
    skeletons the common variables/dimensions of the (output) nc-file.\n
    Input ->
    *nc* : char; output path of nc-file.
    **kwargs ->
    space : class; class where spatial variables are defined.
    sref_name : char; name of the variable storing the CRS.
    Output -> tuple; nc.group storing the nsim-run + xy_tags-dim_indexers.
    """
    xpace = kwargs.get('space', SPACE)
    sref_name = kwargs.get('sref_name', 'spatial_ref')

    # define common dimensions
    nc.createDimension('y', len(xpace.ys))
    nc.createDimension('x', len(xpace.xs))

    """
LOCAL.CRS (netcdf definition)
-----------------------------
Customization of these parameters for your local CRS is relatively easy!.
All you have to do is to 'convert' the PROJ4 (string) parameters of your (local)
projection into CF conventions.
The following links offer you a guide on how to do so,
and the conventions between CF & PROJ4 & WKT:
# https://cfconventions.org/wkt-proj-4.html
# http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#appendix-grid-mappings
# http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#_trajectories
# https://spatialreference.org/
In this case, the PROJ4.string (from the WKT) is:
    pp.CRS.from_wkt(WKT_OGC).to_proj4(); (or) pp.CRS.from_epsg(EPSG).to_proj4()
    '+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +ellps=sphere +units=m +no_defs +type=crs'
which is very similar to the one provided by ISRIC/andresQuichimbo:
    '+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
The amount (and type) of parameters will vary depending on your local CRS. For
instance [http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#lambert-azimuthal-equal-area],
The LAEA (lambert_azimuthal_equal_area) system requieres 4 parameters:
    longitude_of_projection_origin, latitude_of_projection_origin,
    false_easting, & false_northing
which correspond to PROJ4 parameters [https://cfconventions.org/wkt-proj-4.html]:
    +lon_0, +lat_0, +x_0, +y_0
The use of PROJ4 is now being discouraged (https://inbo.github.io/tutorials/tutorials/spatial_crs_coding/).
Neverthelesss, and for now, it still works under this framework to store data
in the local CRS, and at the same time be able to visualize it in WGS84
(via, e.g., https://www.giss.nasa.gov/tools/panoply/) without the need to
transform (and store) local coordinates into Lat-Lon.
[02/08/23] We now use RIOXARRAY to "attach" the CRS, and establish some common
parameters to generate some consistency when reading future? random rain-fields.
    """
# IF FOR SOME REASON YOU'D PREFER TO STORE YOUR DATA IN WGS84...
# COMMENT OUT THE FOLLOWING SECTION & ACTIVATE THE SECTION BELOW (i.e.,):
#    '#~ NETCDF4 definition of "WGS84.CRS" (& grid) ~...'

    #~ NETCDF4 definition of "LOCAL.CRS" (& grid) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    xoid = field.empty_map(xpace.xs, xpace.ys, WKT_OGC)  # empty array
    grid = nc.createVariable(sref_name, 'u1')
    grid.long_name = sref_name
    grid.crs_wkt = xoid.spatial_ref.attrs['crs_wkt']
    grid.spatial_ref = xoid.spatial_ref.attrs['crs_wkt']
    # https://www.simplilearn.com/tutorials/python-tutorial/list-to-string-in-python
    # https://www.geeksforgeeks.org/how-to-delete-last-n-rows-from-numpy-array/
    # grid.GeoTransform = ' '.join(map(str, list(xoid.rio.transform())))
    # # this is apparently the "correct" way to store the GEOTRANSFORM!
    grid.GeoTransform = ' '.join(
        map(str, np.roll(np.asarray(xoid.rio.transform()).reshape(3, 3),
                         shift=1, axis=1)[:-1].ravel().tolist()))  # [:-1] removes last row
    # [start] from CFCONVENTIONS.ORG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    grid.grid_mapping_name = 'lambert_azimuthal_equal_area'
    grid.latitude_of_projection_origin = 5
    grid.longitude_of_projection_origin = 20
    grid.false_easting = 0
    grid.false_northing = 0
    # grid.horizontal_datum_name = 'WGS84'  # (this can also be un-commented!)
    grid.reference_ellipsoid_name = 'sphere'
    # new in CF-1.7 [https://cfconventions.org/wkt-proj-4.html]
    grid.projected_crs_name = 'WGS84_/_Lambert_Azim_Mozambique'
    # [ end ] from CFCONVENTIONS.ORG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [start] from https://publicwiki.deltares.nl/display/NETCDF/Coordinates ~~~
    grid._CoordinateTransformType = 'Projection'
    grid._CoordinateAxisTypes = 'GeoY GeoX'
    # [ end ] from https://publicwiki.deltares.nl/display/NETCDF/Coordinates ~~~
    # storing local coordinates (Y-axis)
    yy = nc.createVariable(
        'projection_y_coordinate', 'i4', dimensions=('y'),
        )
    yy[:] = xpace.ys
    yy.coordinates = 'projection_y_coordinate'
    yy.long_name = 'y coordinate of projection'
    yy._CoordinateAxisType = 'GeoY'
    yy.grid_mapping = sref_name
    yy.units = 'meter'
    # storing local coordinates (X-axis)
    xx = nc.createVariable(
        'projection_x_coordinate', 'i4', dimensions=('x'),
        )
    xx[:] = xpace.xs
    xx.coordinates = 'projection_x_coordinate'
    xx.long_name = 'x coordinate of projection'
    xx._CoordinateAxisType = 'GeoX'
    xx.grid_mapping = sref_name
    xx.units = 'meter'

    # #~ NETCDF4 definition of "WGS84.CRS" (& grid) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # xoid = field.empty_map(xpace.xs, xpace.ys, WKT_OGC)  # empty array
    # grid = nc.createVariable(sref_name, 'int')
    # grid.long_name = sref_name
    # # [start] RIO.XARRAY defaults for WGS84 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # # alternative (once the module RCRS is properly called)
    # # grid.crs_wkt = rcrs.CRS.from_epsg(4326).to_wkt()
    # # grid.spatial_ref = rcrs.CRS.from_epsg(4326).to_wkt()
    # grid.crs_wkt = pp.crs.CRS(4326).to_wkt()
    # grid.spatial_ref = pp.crs.CRS(4326).to_wkt()
    # # the line below ONLY works IF "XOID" was generated in WGS84! (or reprojected to it!)
    # grid.GeoTransform = ' '.join(map(str, list(xoid.rio.transform())))
    # grid.grid_mapping_name = 'latitude_longitude'
    # grid.semi_major_axis = 6378137.
    # grid.semi_minor_axis = 6356752.314245179
    # grid.inverse_flattening = 298.257223563
    # grid.reference_ellipsoid_name = 'WGS 84'
    # grid.longitude_of_prime_meridian = 0.
    # grid.prime_meridian_name = 'Greenwich'
    # grid.geographic_crs_name = 'WGS 84'
    # # [ end ] RIO.XARRAY defaults for WGS84 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # [start] from https://publicwiki.deltares.nl/display/NETCDF/Coordinates ~~~
    # grid._CoordinateAxisTypes = 'Lat Lon'
    # # [ end ] from https://publicwiki.deltares.nl/display/NETCDF/Coordinates ~~~
    # # storing WGS84 coordinates
    # # https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1
    # lat, lon = pp.Transformer.from_proj(
    #     pp.CRS.from_wkt(WKT_OGC).to_proj4(), 'EPSG:4326').transform(
    #         np.meshgrid(xpace.xs, xpace.ys)[0],
    #         np.meshgrid(xpace.xs, xpace.ys)[-1],
    #         zz=None, radians=False
    #         )
    # # (Y-axis)
    # yy = nc.createVariable(
    #     'latitude', 'f8', dimensions=('y', 'x'),
    #     )
    # yy[:] = lat
    # yy.coordinates = 'latitude'
    # yy.long_name = 'latitude coordinate'
    # yy._CoordinateAxisType = 'Lat'
    # yy.grid_mapping = sref_name
    # yy.units = 'degrees_north'
    # # (X-axis)
    # xx = nc.createVariable(
    #     'longitude', 'f8', dimensions=('y', 'x'),
    #     )
    # xx[:] = lon
    # xx.coordinates = 'longitude'
    # xx.long_name = 'longitude coordinate'
    # xx._CoordinateAxisType = 'Lon'
    # xx.grid_mapping = sref_name
    # xx.units = 'degrees_east'

    # store the MASK
    ncmask = nc.createVariable(
        'mask', 'i1', dimensions=('y', 'x'), zlib=True, complevel=9,
        )
    ncmask[:] = xpace.catchment_mask
    ncmask.grid_mapping = sref_name
    ncmask.long_name = 'catchment mask'
    ncmask.description = '1 means catchment or region : 0 is void'
    ncmask.coordinates = f'{yy.getncattr("coordinates")} '\
        f'{xx.getncattr("coordinates")}'

    # store the kmeans/regions
    kmeans = nc.createVariable(
        'regions', 'i1', dimensions=('y', 'x'), zlib=True, complevel=9,
        fill_value=-1,
        )
    # kmeans[:] = -1
    kmeans.grid_mapping = sref_name
    kmeans.long_name = 'regions'
    kmeans.description = 'zero or greater means region : -1 is void'
    kmeans.coordinates = f'{yy.getncattr("coordinates")} '\
        f'{xx.getncattr("coordinates")}'

    return nc, yy.getncattr("coordinates"), xx.getncattr("coordinates")


def nc_file_v(nc, iyear, times, ytag, xtag, **kwargs):
    """
    skeletons the unique sub-group vars/dims of the (output) nc-file.\n
    Input ->
    *nc* : char; output path of nc-file.
    *iyear* : int; subgroup name representing the simulated year
    *times* : np.array; integer array with times (in minutes) since some origin.
    *tag_y* : char; coords-attribute in the Y-axis.
    *tag_x* : dict; coords-attribute in the X-axis.\n
    **kwargs ->
    date_origin : datetime.datetime; DATE_ORIGIN + TIME_ZONE in datetime format
    space : class; class where spatial variables are defined.
    rain_var_n : char; nc-variable name for the rainfall (variable).\n
    Output -> nc.sub_group; nc variable storing the simulated-year.
    """
    dateo = kwargs.get('date_origin', date_origen)
    xpace = kwargs.get('space', SPACE)
    rname = kwargs.get('rain_var_n', RAIN_NAME)

    # define SUB.GROUP and its dimensions
    sub_grp = nc.createGroup(f'{"{:04d}".format(iyear)}')
    sub_grp.createDimension('time', len(times))

    # # define the TIME dimension (& variable)
    nctnam = 'time'  # f'time_{"{:03d}".format(simy + 1)}'  # for < 1000 years
    # storing dates (time-axis)
    timexx = sub_grp.createVariable(
        nctnam, TIMEINT, dimensions=('time'), fill_value=TIMEFIL,
        )
    timexx[:] = times
    timexx.long_name = 'starting time'
    timexx.units = f"{TIME_OUTNC} since {dateo.strftime('%Y-%m-%d %H:%M:%S')}"
    # timexx.units = f"{TIME_OUTNC} since {dateo.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"
    timexx.calendar = 'proleptic_gregorian'  # 'gregorian'
    timexx._CoordinateAxisType = 'Time'
    timexx.coordinates = nctnam

    # define the RAINFALL variable
    # ncvnam = f'year_{iyear}'
    ncvnam = rname
    if RAINFMT[0] == 'f':
        # DOING.FLOATS
        ncvarx = sub_grp.createVariable(
            ncvnam, datatype=f'{RAINFMT}', dimensions=('time', 'y', 'x'),
            zlib=True, complevel=9, least_significant_digit=3, fill_value=np.nan,
            )
    else:
        # DOING.INTEGERS
        ncvarx = sub_grp.createVariable(
            ncvnam, datatype=f'{RAINFMT}', dimensions=('time', 'y', 'x'),
            zlib=True, complevel=9,
            fill_value=np.array(0).astype(f'{RAINFMT}'),  # 0 is filling!
            )
        """
        it'd be tempting to do here:
        ncvarx.scale_factor = SCL
        ncvarx.add_offset = ADD
        nevertheless, doing such a thing here will only cause errors in the\
        variable.values when reading them (back) from the nc.file... so DON'T!!
        """
    ncvarx.precision = PRECISION
    ncvarx.long_name = 'rainfall'
    ncvarx.units = 'mm'
    ncvarx.grid_mapping = nc['spatial_ref'].long_name
    ncvarx.coordinates = f'{ytag} {xtag}'

    # # storing of some XTRA-VARIABLES:
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # e.g., "duration"...
    # ncxtra = sub_grp.createVariable(
    #     'duration', 'f4', dimensions=('t', 'n'),  # ??
    #     zlib=True, complevel=9, fill_value=np.nan,
    #     # fill_value=np.r_[0].astype('u2')),
    #     )
    # ncxtra.long_name = 'storm duration'
    # ncxtra.units = 'minutes'
    # ncxtra.precision = f'{1/60}'  # (1 sec); see last line of 'nc_file_ii'
    # # ncxtra.scale_factor = dur_SCL  # this would've to be estimated
    # # ncxtra.add_offset = dur_ADD  # this would've to be estimated
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # e.g., "sampled_total"...
    # iixtra = sub_grp.createVariable(
    #     'sampled_total', 'f4', dimensions=('n'),  # ??
    #     zlib=True, complevel=9, fill_value=np.nan,
    #     )
    # iixtra.long_name = 'seasonal total from PDF'
    # iixtra.units = 'mm'
    # # iixtra.grid_mapping = nc['spatial_ref'].long_name  # not needed here

    return sub_grp


# %% regionalisation

def regionalisation(file_zon, tag, val, xpace, **kwargs):
    """
    arrange into a dictionary shp-regions.\n
    Input ->
    *file_zon* : char; path to rain-regions shapefile.
    *tag* : char; geoPandas.GeoDataFrame column used as burning values.
    *val* : char; geoPandas.GeoDataFrame column used as numeric lead.
    *xpace* : class; class where spatial variables are defined.\n
    **kwargs ->
    add : int; signed integer to add to k-means mask.\n
    Output -> dict; ...
    """
    magik = kwargs.get('add', -1)

    reg_shp = gpd.read_file(abspath(join(parent_d, file_zon)))
    # transform it into EPSG:42106 & make the buffer
    # https://gis.stackexchange.com/a/328276/127894  (geo series into gpd)
    reg_shp = reg_shp.to_crs(crs=xpace.wkt_prj)  # //epsg.io/42106.wkt
    # k-means for free
    # k_means = region_to_numpy(reg_shp, xpace)
    k_means = masking.shapsterize(
        reg_shp.to_json(), xpace.x_res, xpace.x_res, -1, tag,
        [xpace.bbbox[i] for i in ['l', 'b', 'r', 't']],
        pp.CRS.from_wkt(xpace.wkt_prj).to_proj4(), add=False,
        )
    # plt.imshow(k_means, interpolation='none', cmap='turbo')  # testing
    # numpy masks (for all the unique values but -1)
    # ... 1st transform K-mean into 1s (because of the 0 K-mean);
    # ... then assign 0 everywhere else.
    reg_np = list(map(lambda x: xr.DataArray(k_means).where(
        k_means != x, 1).where(k_means == x, 0).data,
        np.setdiff1d(np.unique(k_means), magik)))
    # as long as the GeoDataFrame comes from "pdfs_", it'll always have 'u_rain'
    output = dict(zip(
        ('mask', 'rain', 'npma', 'kmeans'),
        # (reg_shp, reg_shp['u_rain'], reg_np, k_means)
        (reg_shp, reg_shp[val], reg_np, k_means)
        ))
    return output


# %% reading pdfs

def read_pdfs(*args):
    """
    reads the CSV-file (where PDF's parameters are stored).\n
    Input: optional.\n
    *args ->
    char; path to the csv-file.
    Output -> pd.DataFrame with tabulated pdf-parameters.
    """
    # silly loop so it can take any custom name and don't check it out/testing
    # https://dev.to/trinityyi/how-to-check-if-a-tuple-is-empty-in-python-2ie2
    if not args:
        file_pdf = PDF_FILE
        # append season_&_regions tag to file.name
        fil_tag = f'_{SEASON_TAG}_{NREGIONS}r.csv'
        if fil_tag not in file_pdf:
            file_pdf = file_pdf.replace('.csv', fil_tag)
    else:
        file_pdf = args[0]
    # https://stackoverflow.com/a/58227453/5885810  (import tricky CSV)
    pdfs = pd.read_fwf(abspath(join(parent_d, file_pdf)), header=None)
    pdfs = pdfs[0].str.split(',', expand=True).set_index(0).astype('f8')
    return pdfs


def retrieve_pdf(tag, pd_fs):
    """
    constructs PDFs from parameters.\n
    Input ->
    *tag* : char; label to construct the pdf on ('DATIME_VMF', 'TOTALP_PDF'...).
    *pd_fs* : pd.DataFrame; dataframe with tabulated pdf-parameters.\n
    Output -> REGIONS-list with either constructed scipy-distros or MvMF or rhos.
    """
    subset = pd_fs[pd_fs.index.str.contains(tag)].dropna(
        how='all', axis='columns')
    # necessary 'line' as COPxxx_RHO, MAXINT_PDF, INTRAT_PDF might have Z-bands
    # https://stackoverflow.com/a/50504635/5885810  (escaping \d; & \+)
    # https://stackoverflow.com/a/6400969/5885810  (regex for 1-100)
    # ...in case somebody goes crazy having up to 100 Z-bands!! ('[Z][1-9]' -> otherwise)
    line = subset.index.str.contains(pat=r'[Z]\d{1,2}(?!\d)|100')
    if line.any() and Z_CUTS:
        subset = subset[line]
        name = np.unique(list(zip(*subset.index.str.split('+')))[2])
    else:
        subset = subset[~line]
        name = ['']  # makes "distros" 'universal'
    # https://www.geeksforgeeks.org/python-get-first-element-of-each-sublist/
    first = list(list(zip(*subset.index.str.split('+')))[0])
    # https://stackoverflow.com/a/6979121/5885810   (numpy argsort equivalent)
    # https://stackoverflow.com/a/5252867/5885810
    # https://stackoverflow.com/a/46453340/5885810  (difference between strings)
    sort_id = np.unique(list(map(lambda x: x.replace(tag, ''), first)))
    # grouping by regions
    group = [subset[subset.index.str.contains(f'{i}\\+{tag}')].dropna(
        how='all', axis='columns') for i in sort_id]

    # https://cmdlinetips.com/2018/01/5-examples-using-dict-comprehension/
    # https://blog.finxter.com/how-to-create-a-dictionary-from-two-numpy-arrays/
    # https://stackoverflow.com/a/17615033/5885810  (multiple conds at once)
    # if tag in {'DATIME_VMF', 'DOYEAR_VMF', 'DIRMOV_VMF'}:
    if tag.endswith('_VMF'):
        distros = [
            {A: B for A, B in zip(['alpha', 'phi', 'kappa'], [
                i.to_numpy() for item, i in G.T.iterrows()])} for G in group]
    elif tag.endswith('_RHO') and tag.startswith('COP'):
        distros = [
            {A: B for A, B in zip(name if Z_CUTS else name, [
                i.values.ravel()[0] for item, i in G.iterrows()])} for G in group]
    else:
        distros = [
            {A: B for A, B in zip(name, [
                eval(f"stats.{item.split('+')[-1]}"
                     f"({','.join(i.dropna().astype('str').values.ravel())})")
                for item, i in G.iterrows()])} for G in group]
    return distros


def construct_pdfs(pdframe, **kwargs):
    """
    sets up and fills the global key pdf-parameters for STORM to work.\n
    Input ->
    *pdframe* : pd.DataFrame; dataframe with tabulated pdf-parameters.\n
    **kwargs ->
    tactic : int; strategy to compute STORM (1 via BETPAR; else via COPxxx).\n
    Output: None.
    """
    taktik = kwargs.get('tactic', 1)
    # set this up carefully according to developed STORM (modeling) TACTICs
    core_tag = {
        'TOTALP': 'TOTALP_PDF',
        'AVGDUR': 'AVGDUR_PDF',
        'MAXINT': 'MAXINT_PDF',
        'RADIUS': 'RADIUS_PDF',
        'VELMOV': 'VELMOV_PDF',
        'DIRMOV': 'DIRMOV_VMF',
        'DATIME': 'DATIME_VMF',
        'DOYEAR': 'DOYEAR_VMF'
        }
    tact_tag = {
        'BETPAR': 'BETPAR_PDF'
        } if taktik == 1 else {
        'INTRAT': 'INTRAT_PDF',
        'COPONE': 'COPONE_RHO'
        }
    # https://stackoverflow.com/a/73660961/5885810  (merge dicts)
    _tags = {**core_tag, **tact_tag}

    # defining AND computing GLOBALS (MAXINT, AVGDUR, etc.)
    # https://stackoverflow.com/a/10852003/5885810
    # https://stackoverflow.com/q/423379/5885810  (global variables)
    # https://stackoverflow.com/a/5599313/5885810  (using exec instead of eval)
    # https://stackoverflow.com/a/2083375/5885810  (exec global... weird)
    # https://stackoverflow.com/a/29442282/5885810  (custom raised messages)
    for _, key in enumerate(_tags):
        try:
            # MAXINT = retrieve_pdf('MAXINT_PDF', pdframe)  # e.g.
            exec(f'globals()["{key}"] = retrieve_pdf(_tags["{key}"], pdframe)')
        except IndexError as e:
            raise Exception(f'\n\n+{_tags[key]}+ is not in "{PDF_FILE}".') from e
        # eval(f'print(globals()["{key}"])')

    # each NREGIONS must have the same number of pdfs
    # https://stackoverflow.com/a/12897477/5885810  (unique in list)
    # https://stackoverflow.com/a/18666622/5885810  (speed in list2numpy)
    test = np.array(list(_tags.keys()))
    lens = list(map(len, list(map(eval, test))))
    xtra = np.isin(lens, list(set(lens))[0])  # take the 1st/lowest unique
    assert len(set(lens)) == 1, '\n\nThere are less defined PDFs for '\
        f'{" & ".join(test[xtra])} than for {" & ".join(np.delete(test, xtra))}'\
        f' (in the {NREGIONS} NREGIONS). \nPlease modify the file "{PDF_FILE}"'\
        f' (accordingly) so that each of the NREGIONS has the same number of PDFs.'


# %% sampling

def truncated_sampling(distro, **kwargs):
    """
    sampling (truncated or not) preserving the n-requested.\n
    Input ->
    *distro* : dict; contains a scipy.stats (pdf) frozen infrastructure.\n
    **kwargs ->
    limits : tuple; variable limits to sample within.
    type_l : char; limits nature (either 'prob' or 'var' -> default).
    band : char; key (of the 'distro' dictionary) addressing the frozen pdf.
    n : int; numbers of (random) samples.\n
    Output -> np.array of floats with n-samples.
    """
    limits = kwargs.get('limits', (-np.inf, np.inf))
    timit = kwargs.get('type_l', 'var')
    band = kwargs.get('band', '')
    n = kwargs.get('n', 1)
    # set up useful range from limits
    ulims = list(map(distro[band].cdf, limits)) if timit == 'var' else limits
    # sample via a uniform.PPF
    sample = distro[band].ppf(npr.uniform(low=ulims[0], high=ulims[-1], size=n))
    # # reproducibility...
    # sample = distro[band].ppf(npr.RandomState(npr.SFC64(54321)).uniform(ulims[0], ulims[-1], n))
    # sample = distro[band].ppf(npr.RandomState(npr.PCG64(2001)).uniform(ulims[0], ulims[-1], n))
    # sample = distro[band].rvs(size=n, random_state=npr.RandomState(npr.Philox(12345)))
    # sample = distro[band].rvs(size=n, random_state=npr.RandomState(npr.PCG64DXSM(1337)))
    return sample


def faster_sampling(distro, **kwargs):
    """
    faster sampling (truncated or not) preserving the n-requested.\n
    Input ->
    *distro* : dict; contains a scipy.stats (pdf) frozen infrastructure.\n
    **kwargs ->
    limits : tuple; variable limits to sample within.
    band : char; key (of the 'distro' dictionary) addressing the frozen pdf.
    n : int; numbers of (random) samples.\n
    Output -> np.array of floats with n-samples.
    """
    limits = kwargs.get('limits', (-np.inf, np.inf))
    band = kwargs.get('band', '')
    n = kwargs.get('n', 1)
    # sample via a .rvs
    sample = []
    while n > 0:
        subple = distro[band].rvs(size=n)
        # chopping into limits
        subple = subple[(subple >= limits[0]) & (subple <= limits[-1])]
        n = n - len(subple)
        sample.append(subple)
    return np.concat(sample)


def copula_sampling(copula, dist_one, dist_two, **kwargs):
    """
    samples randomly (n-size) from a copula, conditional to two distros.\n
    Input ->
    *copula* : dict; contains a np.float (i.e., rho) per elevation band
    *dist_one* : dict; contains a scipy.stats (pdf) frozen infrastructure
    *dist_two* : dict; contains a scipy.stats (pdf) frozen infrastructure.\n
    **kwargs ->
    band : char; key (of the 'qant' dictionary) addressing the frozen pdf.
    n : int; numbers of (random) samples.\n
    Output -> tuple; containing np.floats for dist_one-samples (first) and \
        dist_two-samples (last).
    """
    band = kwargs.get('band', '')
    n = kwargs.get('n', 1)
    # https://stackoverflow.com/a/12575451/5885810  (1D numpy to 2D)
    # 'reshape' allows for n==1 sampling
    # (-1, 2) cause 'GaussianCopula' always give 2 columns (BI-variate copula)
    cop_space = GaussianCopula(corr=copula[band], k_dim=2).rvs(nobs=n).reshape(-1, 2)
    # # for reproducibility
    # cop_space = GaussianCopula(corr=copula[band], k_dim=2).rvs(nobs=n,
    #     random_state=npr.RandomState(npr.PCG64(20220608))).reshape(-1, 2)

    # # return pd.Series instead for easy concatenation (and shuffling) later
    # var_one = dist_one[band].ppf(cop_space[:, 0])
    # var_two = dist_two[band].ppf(cop_space[:, 1])
    var_one = pd.Series(dist_one[band].ppf(cop_space[:, 0]), name=band)
    var_two = pd.Series(dist_two[band].ppf(cop_space[:, 1]), name=band)
    return var_one, var_two


class scentres:

    def __init__(self, shape, size, **kwargs):
        """
        performs a poisson.point.process inside a shp [fast approach].\n
        Input ->
        *shape* : pd.Series; pandas series with GEOMETRY.
        *size* : int; number of point to sample.\n
        **kwargs ->
        realizations : int; number of realizations.
        conditioning : bool; not exact number of *size* samples?.
        aspp : bool; as pandas.DataFrame?.
        out_real : int; index of the realization to output.\n
        Output -> a class random sampled (spatial) points.
        """
        self.shp_series = shape
        self.size = size
        self.n_real = kwargs.get('realizations', 1)
        self.condition = kwargs.get('conditioning', False)
        self.aspp = kwargs.get('aspp', True)
        self.what_real = kwargs.get('out_real', 0)
        self._all_real = self.rvs()
        self.samples = self._all_real.realizations[self.what_real] if not\
            self.aspp else self._all_real.realizations[self.what_real].points.to_numpy()

    """
    spatial.random.sampling (inside the SHP)
    simulate a Complete Spatial Randomness (CSR) process inside a SHP.
    (maybe it'd be a good idea to compute many realizations, and then
     sample from them iteratively)?
    "asPP"==True  -> generates a point pattern pandas (easy visualization)
    "asPP"==False -> generates a point series numpy
    "conditioning"==True  -> simulates a lamda-conditioned CSR  (variable SIZE)
    "conditioning"==False -> simulates a N-conditioned CSR      (exactly  SIZE)
    !!the more REALIZATIONS and more POINTS, the slower it gets!!
    """

    def rvs(self,):
        """
        random sampling.\n
        Input: none.\n
        Output -> np.array; 2D-numpy with X-Y (column) coordinates.
        """
        # transform SHAPELY into PYSAL
        tmp = [np.array(
            self.shp_series['geometry'].geoms[i].exterior.coords.xy).T.tolist()
            for i in range(len(self.shp_series['geometry'].geoms))]
        sal_shp = ps.cg.Polygon(tmp)
        # (sal_shp.len, sal_shp.holes)
        # transform the PYSAL into a WINDOW
        wndw_bffr = Window(sal_shp.parts)

        samples = PoissonPointProcess(
            window=wndw_bffr, n=self.size, samples=self.n_real,
            conditioning=self.condition, asPP=self.aspp
            )
        # # 565 ms ± 27.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # # visualization
        # samples.realizations[0].plot(window=True, title='point pattern')  # if asPP=True
        # PointPattern(samples.realizations[0]).plot(window=False, hull=False,
        #                                            title='point series')  # if asPP=False
        return samples

    def plot(self, **kwargs):
        prnt = kwargs.get('file', None)
        region = kwargs.get('region', self.shp_series['geometry'])
        center = kwargs.get('cents', self.samples)
        edgec = kwargs.get('edge_col', 'xkcd:amethyst')
        mtype = kwargs.get('mark_type', 'P')
        msize = kwargs.get('mark_size', 37)
        mcol = kwargs.get('mark_col', 'xkcd:dark blue')
        # plot
        fig = plt.figure(figsize=(10, 10), dpi=300)
        # ax = plt.axes(projection=ccrs.epsg(42106))  # EPSG code not official!
        ax = plt.axes()
        ax.set_aspect(aspect='equal')
        # https://stackoverflow.com/a/2176591/5885810  # ticks/labels invisible
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor(None)
        fig.tight_layout(pad=0)
        # https://gis.stackexchange.com/a/426634/127894  (plot shapely.Polygons)
        gpd.GeoSeries(region).plot(
            edgecolor=edgec, alpha=1., zorder=2,
            linewidth=.77, ls='dashed', facecolor='None', ax=ax,
            )
        plt.scatter(center[:, 0], center[:, 1], edgecolors='none', color=mcol,
                    marker=mtype, s=msize,)
        # show
        plt.show() if not prnt else\
            plt.savefig(prnt, bbox_inches='tight', pad_inches=0.01,
                        facecolor=fig.get_facecolor())


class scentros:

    def __init__(self, shape, size):
        """
        performs a poisson.point.process inside a shp [slow approach].\n
        Input ->
        *shape* : pd.Series; pandas series with GEOMETRY.
        *size* : int; number of point to sample.\n
        Output -> a class random sampled (spatial) points.
        """
        self.shp_series = shape
        self.samples = random.poisson(self.shp_series['geometry'], size=size)

    """
    # FROM: https://stackoverflow.com/a/69630606/5885810  (random poitns in SHP)
    alternative also coming from libpysal. not very fast as of 25/09/24, though.
    It potentially also allows for some 'Clustered Point Pattern' generation,
    throughout the method "random.cluster_poisson(...".
    """


# %% timing

def wet_days(year, **kwargs):
    """
    defines the days of the (wet)-season to sample from (given a "seed" year).\n
    Input ->
    *year* : int; modified (or not) SEED_YEAR.\n
    **kwargs ->
    season_tag : char; three-letter season tag ('MAM', 'JJAS' or 'OND').\n
    Output -> tuple; number of months in the season (first) & list of \
        start & end 'datetime.datetime(s)' of the season (last).
    """
    tag = kwargs.get('season_tag', SEASON_TAG)
    # establish the SEASONAL-dict
    sdic = {'MAM': [3, 4, 5], 'JJAS': [6, 7, 8, 9], 'OND': [10, 11, 12]}
    # monthly duration for season(al)-[tag] (12 months in a year)
    nonths = sdic[SEASON_TAG][-1] - sdic[SEASON_TAG][0] + 1
    # construct the date.times
    ini_date = datetime(year=year, month=sdic[SEASON_TAG][0], day=1)
    datespool = [ini_date, ini_date + relativedelta(months=nonths)]
    datespool_p = [datespool[0], datespool[-1] - relativedelta(seconds=1)]
    # extract Day(s)-Of-Year(s)
    DOY_POOL = list(map(lambda d: d.timetuple().tm_yday, datespool_p))
    return nonths, datespool


def base_round(stamps, **kwargs):
    """
    rounds numbers or time.stamps to either floor or nearest specific T_RES/base.\n
    Input ->
    *stamps* : np.array; float numpy (also representing seconds since some origin).\n
    **kwargs ->
    base : int; rounding (temporal) resolution.
    time_tag : char; string indicating the base resolution.
    time_dic : dic; dictionary containing the equivalences of 'tags' in base 60.
    base_system: int; conversion constant for the time_dic-system (default: 60).
    method : char; rounding method (either 'floor' or 'nearest').
    round : int; significative digits/places for rounding.\n
    Output -> rounded numpy (to custom resolution/base, i.e., T_RES).
    """
    base = kwargs.get('base', T_RES)
    time_dic = kwargs.get('time_dic', TIME_DICT_)
    time_tag = kwargs.get('time_tag', TIME_OUTNC)
    s_base = kwargs.get('base_system', 60)  # sexagesimal
    kase = kwargs.get('method', 'floor')
    ndec = kwargs.get('round', 3)
    # https://stackoverflow.com/a/2272174/5885810
    # https://stackoverflow.com/a/18666678/5885810
    # update 'base': on a normal day "time_dic[time_tag]" == 1/60 (& 1/60*60==1)
    # ... that's why 'base' is unaltered, and the function becomes "universal"
    base = base * time_dic[time_tag] * s_base
    if kase == 'floor':
        iround = (base * (np.ceil(stamps / base) - 1))  # .astype(TIMEINT)
    elif kase == 'nearest':
        # iround = (base * (stamps / base).round(0)).round(ndec)  # .astype(TIMEINT)
        iround = (base * np.round(stamps / base, 0)).round(ndec)  # .astype(TIMEINT)
    else:
        raise TypeError("Wrong method passed!\n"
                        "Pass 'floor' or 'nearest' to the 'method' argument.")
    return iround


def slice_time(time_raw, time_rnd, s_dur, **kwargs):
    """
    splits storm duration into discrete/regular time slices.\n
    Input ->
    *time_raw* : np.array; float numpy representing seconds since some origin.
    *time_rnd* : np.array; float numpy of base-rounded seconds since some origin.
    *s_dur* : np.array; float numpy of storm durations (in hours).\n
    **kwargs ->
    time_tag : char; string indicating the time-resolution of storm durations.\n
    Output -> list of np.arrays containing time-slices of storm durations.
    """
    time_tag = kwargs.get('time_tag', 'hours')
    # s_dur in minutes
    s_dur = s_dur / (TIME_DICT_[time_tag] * 60)
    date_diff = time_raw + -1 * time_rnd
    # how much time to the right (from STARTING.TIME) you have in the 1st slice
    left_dur = 1 - date_diff / (T_RES * TIME_DICT_[TIME_OUTNC] * 60)
    # how many complete-and-remaining tiles you have for slicing
    cent_dur = s_dur / T_RES - left_dur
    # negatives imply storm.duration smaller than slice [so update 1st slice]
    left_dur[cent_dur < 0] = s_dur[cent_dur < 0] / T_RES
    # extract the number of complete-centered slices
    cent_int = cent_dur.astype(TIMEINT)
    # remainings of the complete-centered slices is what goes to the last slice
    righ_dur = cent_dur - cent_int
    righ_dur[righ_dur < 0] = np.nan  # remove negatives
    # establish/repeat the number of centered-whole slices
    # https://stackoverflow.com/a/3459131/5885810
    cent_int = list(map(lambda x, y: [x] * y, [1] * len(cent_int), cent_int))
    # join LEFT, CENTER, and RIGHT slices
    sfactors = list(map(np.hstack, np.stack(
        [left_dur, np.array(cent_int, dtype='object'), righ_dur],
        axis=1).astype('object')))
    # remove NANs, and apply rainfall.scalability factor
    # '/ 60' (60 mins in 1h) because T_RES -> minutes & S_DUR -> mm/h [ALWAYS!]
    sfactors = list(map(lambda x: x[~np.isnan(x)] * T_RES / 60, sfactors))
    return sfactors


def quantum_time(doy_par, tod_par, DUR_S, date_pool, n, **kwargs):
# doy_par=DOYEAR[nreg]; tod_par=DATIME[nreg]; n=NUM_S
    """
    samples datetimes and quatize them into packs of storm duration(s).\n
    Input ->
    *doy_par* : dic; vonMises-Fisher mixture-parameters for Day-of-Year.
    *tod_par* : dic; vonMises-Fisher mixture-parameters for Time-of-Day.
    *DUR_S* : np.array; float numpy of storm durations (in hours).
    *date_pool* : list of start & end 'datetime.datetime(s)' of the season.
    *n* : int; sample size.\n
    **kwargs ->
    t_res : int; temporal resolution.
    time_tag : char; string indicating the base resolution.
    time_dic : dic; dictionary containing the equivalences of 'tags' in base 60.
    base_system: int; conversion constant for the time_dic-system (default: 60).
    tz : char; standarized name of the (local) time zone.
    t_format : char; string indicating the output format ('u4', 'f8'... etc).\n
    Output -> tuple; list-quantized hourly resolutions & \
        their xploded equivalent (date)times.
    """
    base = kwargs.get('t_res', T_RES)
    time_dic = kwargs.get('time_dic', TIME_DICT_)
    time_tag = kwargs.get('time_tag', TIME_OUTNC)
    s_base = kwargs.get('base_system', 60)  # sexagesimal
    time_zon = kwargs.get('tz', TIME_ZONE)
    time_int = kwargs.get('t_format', TIMEINT)

    # computing DOY
    M = n
    all_dates = []
    while M > 0:
        cs_day = circular(doy_par,)
        doys = cs_day.samples(M, data_type='doy') - 1
        # cs_day.plot_samples(data=doys, data_type='doy', bins=40)  # plotting
        # # no necessary as there isn't negative dates produced
        # doys = doys[doys > 0]  # negative doys?? (do they belong to jan/dec?)
        # into actual dates
        dates = list(map(lambda d: datetime(year=date_pool[0].year, month=1, day=1)
                         + relativedelta(yearday=int(d)), doys.round(0)))
        sates = pd.Series(dates)  # to pandas
        # chopping into limits
        sates = sates[(sates >= date_pool[0]) & (sates <= date_pool[-1])]
        M = len(dates) - len(sates)
        all_dates.append(sates)
    all_dates = pd.concat(all_dates, ignore_index=True)
    # shuffle the dates (so later the selection can be done left-to-right)
    # # https://stackoverflow.com/a/72040563/5885810  (shuffle pandas)
    # pd.Series(np.random.permutation(all_dates))
    np.random.RandomState(seed=None).shuffle(all_dates)
    # np.random.RandomState(
    #     seed=np.random.randint(0, 4294967295, 1, dtype=np.int64)
    #     ).shuffle(all_dates)
    # # https://github.com/DLR-RM/stable-baselines3/issues/1579#issuecomment-1611892556

    # computing TOD
    cs_tod = circular(tod_par,)
    times = cs_tod.samples(n, data_type='tod')
    # cs_tod.plot_samples(data=times, data_type='tod', bins=40)  # plotting
    # https://stackoverflow.com/a/50062101/5885810

    # SECONDS since DATE_ORIGIN
    stamps = np.asarray(
        list(map(
            lambda d, t: (d + timedelta(hours=t) - date_origen).total_seconds(),
            all_dates.dt.tz_localize(time_zon), times
            ))
        )
    # # https://stackoverflow.com/a/67105429/5885810  (chopping milliseconds)
    # stamps = np.asarray(list(map(lambda d, t: (
    #     d +timedelta(hours=t)).isoformat(timespec='seconds'), dates, times)))
    stamps = stamps * time_dic[time_tag]  # scaled to output.TIMENC.res

    # round starting.dates to nearest.floor T_RES
    rates = base_round(stamps)
    # turn the DUR_S into discrete time.slices
    s_cal = slice_time(stamps, rates, DUR_S)
    # xploding of discrete timestamps (per storm.cluster)
    tres_up = base * time_dic[time_tag] * s_base
    mates = np.concatenate(list(map(lambda r_s, i_s: np.arange(
        start=r_s, stop=r_s + tres_up * len(i_s), step=tres_up),
        rates, s_cal))).astype(time_int)

    return mates, s_cal


# %% raster

def moving_storm(dir_par, vel_par, stridin, centres, **kwargs):
# dir_par=DIRMOV[nreg]; vel_par=VELMOV[nreg]; stridin=STRIDE; centres=CENT.samples
    """
    samples storm direction and velocity; and moves storm initial centres along.\n
    Input ->
    *dir_par* : dic; vonMises-Fisher mixture-parameters for storm-direction.
    *vel_par* : dict; contains a scipy.stats (pdf) frozen infrastructure.
    *stridin* : list; numpys list with time-quantization per storm.
    *centres* : numpy; 2D-numpy with the X-Y's of storm initial centers.\n
    **kwargs ->
    speed_lim : tuple; variable limits to sample within.
    speed_stat : char; numpy stat to apply to an array.\n
    Output -> (list of?) 2D-numpy array of X-Y's displaced/moved storm centres.
    """
    s_lims = kwargs.get('speed_lim', (-np.inf, np.inf))
    # s_stat = kwargs.get('speed_stat', None)
    s_stat = kwargs.get('speed_stat', "transpose")

    # sampling storm direction (in m/s)
    i_lens = list(map(len, stridin))
    wspeed = faster_sampling(vel_par, limits=s_lims, n=np.sum(i_lens))
    wspeed = np.split(wspeed, np.array(i_lens).cumsum()[:-1])
    # storm direction (already as azimuth)
    cs_dir = circular(dir_par,)
    azimut = cs_dir.samples(len(i_lens), data_type='dir')

    # displace the storm_centres
    # pad_i = list(map(lambda x: np.concat(([0], x[:-1])), stridin))
    pad_i = list(map(lambda x: np.concat(([0], x[:-1])) * 3600, stridin))
    # update 'wspeed' to 's_stat'
    # wspeed = wspeed if s_stat is None else list(map(eval(f'np.{s_stat}'), wspeed))
    wspeed = list(map(eval(f'np.{s_stat}'), wspeed))

    stride = list(map(np.cumsum, list(map(np.multiply, wspeed, pad_i))))  # -> in meters!!
    # # these "deltas" are computational faster (by 3x) than:
    # # deltax = list(map(lambda s, a: s * 1000 * np.cos(a), stride, azimut))
    # # ... but be mindful that strictly speaking STRIDE is the one to "* 1000"!
    # # "* 1000" because the velocity is in m/s but the reference.grid is in km!
    # deltax = list(map(np.multiply, stride, np.cos(azimut) * 1000))  # (in km!)
    # deltay = list(map(np.multiply, stride, np.sin(azimut) * 1000))  # (in km!)
    deltax = list(map(np.multiply, stride, np.cos(azimut) * 1))  # (in meters!)
    deltay = list(map(np.multiply, stride, np.sin(azimut) * 1))  # (in meters!)

    # updated-and-aggregated centers
    x_s = list(map(np.add, centres[:, 0], deltax))
    y_s = list(map(np.add, centres[:, 1], deltay))

    # put them in the same format as input
    n_cent = np.stack((np.concatenate(x_s), np.concatenate(y_s)), axis=1)
    # if output in a list-of-numpys structured
    n_cent = np.split(n_cent, np.array(i_lens).cumsum()[:-1])
    return n_cent


# # https://gis.stackexchange.com/a/267326/127894  (get EPSG/CRS from raster)
# from osgeo import osr
# tIFF = gdal.Open(abspath(join(parent_d, DEM_FILE)))
# tIFF_proj = osr.SpatialReference(wkt=tIFF.GetProjection()).GetAttrValue('AUTHORITY', 1)
# tIFF = None

def last_ring(radii, centres, **kwargs):
# radii=RADII[0]; centres=M_CENT[0]
    """
    creates a circular polygon of given radius and center.\n
    Input ->
    *radii* : numpy; float numpy of storm-radius (in km).
    *centres* : numpy; 2D-numpy with the X-Y's of storm centers.\n
    **kwargs ->
    scaling_dis : float; scaling factor between radius-units and m (1000m==1km).
    base_radius : float; minimum (base) radius (in km).
    nonsense : int; factor to tailor-suit the segment resolution.\n
    Output -> geopandas.GeoDataFrame with circular polygons of storm-max-radii.
    """
    scal = kwargs.get('scaling_dis', 1e3)
    brad = kwargs.get('base_radius', minmax_radius)
    nons = kwargs.get('nonsense', 2)
    # # slower approach
    # ring_last = list(map(lambda c, r: gpd.GeoDataFrame(
    #     geometry=gpd.points_from_xy(x=[c[0]], y=[c[1]]).buffer(
    #         # r * scal, resolution=int((3 if r < 1 else 2)**np.ceil(r / 2)),
    #         r * scal, resolution=np.ceil(r / brad) + nons),
    #     crs=WKT_OGC), centres, radii))
    geom = gpd.points_from_xy(x=centres[:, 0], y=centres[:, 1])
    geom = geom.buffer(radii * scal, resolution=(np.ceil(radii / brad) + nons).mean())
    # geom = geom.buffer(radii * scal, resolution=8*2)
    ring_last = gpd.GeoDataFrame(geometry=geom, crs=WKT_OGC)
    # "resolution": number of segments in which 1/4.of.a.circle is divided into.
    # (now) it depends on the RADII/minmax_radius; the larger a circle the higher its resolution.
    return ring_last


def lotr(radius, decay, i0, lapse, centre, **kwargs):
# j = np.argmax(list(map(len, time_idx.values())))
# radius=RADII[j]; decay=BETA[j]; i0=MAX_I[j]; lapse=reduce(iconcat, STRIDE, [])[j]; centre=M_CENT[j]
    """
    creates circular rain.rings evenly spaced from the storm.centre outwards.\n
    Input ->
    *radius* : numpy; float numpy of storm-radius (in km).
    *decay* : numpy; float numpy of radial-decay (in 1/km).
    *i0* : numpy; float numpy of maximum storm intesity (in mm/h).
    *lapse* : numpy; float numpy of the actual raining time-lapse (in h).
    *centre* : numpy; 2D-numpy with the X-Y of storm center.\n
    **kwargs ->
    scaling_dis : float; scaling factor between radius-units and m (1000m==1km).
    res : int; number of segments in which a circle.quadrant is divided into.
    dot_size : float; circle's radius emulating the storm centre's point/dot.
    sep_ring : float; separation (in km) between rainfall rings.\n
    Output -> geopandas.GeoDataFrame with linestings of circular.rain from centre.
    """
    scal = kwargs.get('scaling_dis', 1e3)
    res = kwargs.get('res', 13)
    sdot = kwargs.get('dot_size', 0.15)
    sep = kwargs.get('sep_ring',  minmax_radius * (2) + .1)  # 10.1
    # c_radii = np.append(np.arange(radius, sdot, - sep), sdot)
    c_radii = np.concatenate(
        (np.arange(radius, sdot, - sep), np.array([sdot])))
    # rainfall [in mm] for every c_radii
    c_rain = i0 * lapse * np.exp(-2 * decay**2 * c_radii**2)  # in mm!!

    # buffer_strings
    # https://www.knowledgehut.com/blog/programming/python-map-list-comprehension
    # https://stackoverflow.com/a/30061049/5885810  (map nest)
    # .boundary gives the LINESTRING element
    geom = gpd.points_from_xy(x=[centre[0]], y=[centre[1]])
    geom = geom.buffer(c_radii * scal, resolution=res).boundary
    rain_ring = gpd.GeoDataFrame({'rain': c_rain,'geometry':geom}, crs=WKT_OGC)
    # rain_ring.iloc[0].geometry  # rain_ring.iloc[-1].geometry
    return rain_ring


def rasterize(ring_set, outer_ring, **kwargs):
# j=363; ring_set=c_ring[j]; outer_ring=last_r[j]; xpace=SPACE
    """
    rasterize linerings/polygons and interpolate rainfall (between rings).\n
    Input ->
    *ring_set* : geopandas.GeoDataFrame; linerings geometry with rain.
    *outer_ring* : pandas.Series; polygon geometry with outermost (rain) ring.
    **kwargs ->
    space : class; class where spatial variables are defined.\n
    Output -> 2D-numpy of floats representing a circular storm.
    """
    xpace = kwargs.get('space', SPACE)

    # burn the 'ring_set' (for one storm-centre)
    fall = masking.shapsterize(
        ring_set.to_json(), xpace.x_res, xpace.x_res, 0, 'rain',
        [xpace.bbbox[i] for i in ['l', 'b', 'r', 't']],
        pp.CRS.from_wkt(xpace.wkt_prj).to_proj4(), outputType=gdal.GDT_Float32,
        )
    # plt.imshow(fall, interpolation=None, origin='upper', cmap='turbo')

    # burn the mask (create a GEOPANDAS from a SHAPELY so you can JSON.it)
    out_frame = gpd.GeoDataFrame(geometry=outer_ring)
    out_frame['burn'] = 1
    # https://stackoverflow.com/a/51520122/5885810
    mask = masking.shapsterize(
        # gpd.GeoDataFrame({'burn': [1], 'geometry': outer_ring}).to_json(),
        out_frame.to_json(), xpace.x_res, xpace.x_res, 0, 'burn',
        [xpace.bbbox[i] for i in ['l', 'b', 'r', 't']],
        pp.CRS.from_wkt(xpace.wkt_prj).to_proj4(), add=False,
        # outputType=gdal.GDT_Int16,
        )

    # re-touching the mask to do a proper interpolation
    mask[fall != 0.] = 0
    # everything that is 1 is interpolated
    fill.fillnodata(np.ma.array(fall, mask=mask), mask=None,
                    max_search_distance=4., smoothing_iterations=2)
    # # plotting checking:
    # zlice = (slice(330, 410), slice(70, 160))
    # xr.DataArray(fall[zlice]).plot(cmap='gist_ncar_r', levels=22, vmax=0.525,)
    # xr.DataArray(fall[zlice]).plot(cmap='gist_ncar_r', robust=True, vmax=21,)
    # fall[(slice(345, 375), slice(132, 150))].round(2)

    # pass filter ('gaussian' slightly faster than 'uniform')
    # f_filter = uniform_filter(fall, size=3, mode='constant', cval=0)
    f_filter = gaussian_filter(fall, sigma=2, mode='constant', cval=0, radius=1)
    # because filtering averages.down the field, select maxima between...
    ok_field = np.maximum(fall, f_filter)

    # # plot testing: (slices subjective to the data you're testing here)
    # fig = plt.figure(dpi=150)
    # xs = np.arange(155, 215, 1)
    # xv = (slice(44, 45), slice(155, 215))
    # xs = np.arange(167, 180, 1)
    # xv = (slice(25, 26), slice(167, 180))
    # plt.step(xs, fall[xv].ravel(), where='mid', color='r', lw=2, zorder=3)
    # plt.step(xs, ok_field[xv].ravel(), where='mid', color='b', lw=4, zorder=1)
    # plt.show()

    return ok_field


def rain_cube(c_ring, last_r, t_stamp, np_mask, **kwargs):
# t_stamp=list(time_idx.keys()); space=SPACE; max_=iMAX
    """
    creates a rainfall cube (potentially) with regional rain already achieved.\n
    Input ->
    *c_ring* : tuple; geopandas.GeoDataFrame linerings geometry with rain.
    *last_r* : tuple; pandas.Series; polygon geometry with outermost (rain) ring.
    *t_stamp* : list; numpy integers representing time-steps since origin.
    *np_mask* : numpy; 2D-numpy with 1's representing the rain-region.\n
    **kwargs ->
    space : class; class where spatial variables are defined.
    max_val : float; maximum value for rainfall allowed.\n
    Output -> tuple; xarray.DataArray(s) with rounded rainfall (first) and \
        total rainfall (within n-region) for every time-stamp (last).
    """
    space = kwargs.get('space', SPACE)
    max_ = kwargs.get('max_val', iMAX)

    tot_pix = np_mask.sum()  # pixels in mask
    # create empty xarray
    void = xr.DataArray(
        data=np.empty((len(c_ring), len(space.ys), len(space.xs)), dtype=RAINFMT),
        dims=['time', 'y', 'x'],
        coords={'time': t_stamp, 'y': space.ys, 'x': space.xs},
        attrs={
            'units': 'mm',
            # '_FillValue': -99.99,
            '_FillValue': 0,
            }
        )
    # create empty array to store sums
    suma = []
    # fill the void.array with rasterize rainfall
    for i in range(void.shape[0]):
        tmp_rain = rasterize(c_ring[i], last_r[i],)
        tmp_rain = base_round(tmp_rain, method='nearest', base=PRECISION)  # round=4
        tmp_rain[tmp_rain > max_] = max_  # capping above maxima
        # only keep rainfall inside the mask
        tmp_rain[np_mask == 0] = 0.
        # compute sum here, before changing to INTeger
        suma.append(tmp_rain.sum() / tot_pix)
        # now do the INTransformation
        tmp_rain = ((tmp_rain - ADD) / SCL).round()
        void[i, :, :] = tmp_rain  # .astype(RAINFMT)
    # only keep rainfall larger than zero (i.e., 1 INT)
    void = void.where(void > 1, 0)
    # temporal aggregation
    """
    dividing by TOT_PIXimplies REACHING the MEAN (in the stopping criterium).
    if one wants the 'granular' MEDIAN, something else has to be thought about!
    """
    suma_ = xr.DataArray(data=np.array(suma), coords={'time': void['time']},)
    # suma = (void.sum(dim=('x', 'y')) * SCL + ADD) / tot_pix
    return void, suma_


def rain_cube_dask(c_ring, last_r, t_stamp, np_mask, **kwargs):
# t_stamp=list(time_idx.keys()); space=SPACE; max_=iMAX
    """
    creates a rainfall cube (potentially) with regional rain already achieved.\n
    Input ->
    *c_ring* : tuple; geopandas.GeoDataFrame linerings geometry with rain.
    *last_r* : tuple; pandas.Series; polygon geometry with outermost (rain) ring.
    *t_stamp* : list; numpy integers representing time-steps since origin.
    *np_mask* : numpy; 2D-numpy with 1's representing the rain-region.\n
    **kwargs ->
    space : class; class where spatial variables are defined.
    max_val : float; maximum value for rainfall allowed.\n
    Output -> tuple; xarray.DataArray(s) with rounded rainfall (first) and \
        total rainfall (within n-region) for every time-stamp (last).
    """
    space = kwargs.get('space', SPACE)
    max_ = kwargs.get('max_val', iMAX)

    tmp_mask = array.from_array(np_mask, chunks='auto',)
    tot_pix = np_mask.sum()  # pixels in mask
    # # create empty dask xarray
    # vals = delayed(np.zeros)(len(c_ring) * len(space.ys) * len(space.xs))
    # vaid = array.from_delayed(value=vals, shape=(len(c_ring), len(space.ys),
    #                                              len(space.xs)), dtype=RAINFMT)
    # # vaid.visualize()

    # create empty array to store sums
    suma = []
    dain = []
    for i in np.r_[0:5]:  # i=0
        tmpslice = rasterize(c_ring[i], last_r[i],)
        tmpslice = array.from_array(tmpslice.astype('f4'), chunks='auto',) # chunks='0.09 MiB',
        # my_ufunc  = array.gufunc(base_round, signature='()->()',
        #                          output_dtypes='f4', vectorize=False,)
        # tmpslice = my_ufunc(tmpslice, method='nearest', base=PRECISION,)
        tmpslice = array.apply_gufunc(
            base_round, '()->()', tmpslice, output_dtypes='f4',
            vectorize=False, method='nearest', base=PRECISION,
            )
        tmpslice[tmpslice > max_] = max_  # capping above maxima
        # only keep rainfall inside the mask
        tmpslice[tmp_mask == 0] = 0.
        # compute sum here, before changing to INTeger
        suma.append(tmpslice.sum().compute() / tot_pix)
        # now do the INTransformation
        tmpslice = ((tmpslice - ADD) / SCL)
        tmpslice = array.apply_gufunc(np.round, '()->()', tmpslice,
                                      output_dtypes=RAINFMT, vectorize=True,)
        # only keep rainfall larger than zero (i.e., 1 INT)
        tmpslice[tmpslice == 1] = 0
        # tmpslice.visualize()
        # tmpslice.compute()
        dain.append(tmpslice)
    dain = array.stack(dain, axis=0)
    # dain.visualize()
    """
    dividing by TOT_PIXimplies REACHING the MEAN (in the stopping criterium).
    if one wants the 'granular' MEDIAN, something else has to be thought about!
    """
    suma_ = xr.DataArray(data=np.array(suma), coords={'time': void['time']},)
    # suma = (void.sum(dim=('x', 'y')) * SCL + ADD) / tot_pix
    return void, suma_


# %% main loop

def loop(train, mask_shp, np_mask, nsim, simy, nreg, mlen, upd_max, maxima, date_pool):
# train=reg_tot; mask_shp=region_s['mask'].iloc[nreg]; np_mask=region_s['npma'][nreg]
    """
    calls children-functions to compute storms until seasonal rain is reached.\n
    Input ->
    *train* : float; total seasonal rainfall to reach.
    *mask_shp* : pd.Series; pandas series with GEOMETRY.
    *np_mask* : numpy; 2D-numpy with 1's representing the rain-region.
    *nreg* : int; rain-region index.
    *nsim* : int; index/ordinal indicating the simulation.
    *simy* : int; index/ordinal indicating the year under simulation.
    *mlen* : int; number of months in the season
    *upd_max* : float; maximum rainfall intensity (mm/h) allowed.
    *maxima* : int; maximum value (in RAINFMT) (to clip uX-integers).
    *date_pool* : list of start & end 'datetime.datetime(s)' of the season.\n
    **kwargs ->
    Output -> tuple; xarray with seasonal rainfal for a region (first); \
        xarray with total rainfall per time-step (second); \
            float indicating the seasonal total reached (third).
    """
    # for the HAD we assume (initially) there's ~6 storm/day; then
    # ... we continue 'half-ing' the above seed
    # 30*?? ('_SF' runs); 30*?? ('_SC' runs); 30*6?? ('STANDARD' runs)
    NUM_S = 30 * tunnin * mlen
    CUM_S = 0
    contar_int = 0

    lrain = [None]
    lsuma = [None]

    # until total rainfall is reached or no more storms to compute!
    while CUM_S < train and NUM_S >= 2:
    # while contar_int < 1 and NUM_S >= 2:  # does the cycle 1x maximum!
        collect()
#%%
        # sample random storm centres
        CENT = scentres(mask_shp, NUM_S)  # CENT.plot()
        # # 561 ms ± 10.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # SENT = scentros(mask_shp, NUM_S)
        # # 1.07 s ± 29.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # CENT.plot(region=SENT.shp_series['geometry'], cents=SENT.samples)

        # to 'hours' because the AVGDUR PDF is in hours
        # dur_lim = np.array([T_RES, MAX_DUR]) * TIME_DICT_[TIME_OUTNC]
        dur_lim = np.array([MIN_DUR, MAX_DUR]) * TIME_DICT_[TIME_OUTNC]
        DUR_S = faster_sampling(AVGDUR[nreg], limits=dur_lim, n=NUM_S)
        # "faster_sampling" is indeed FASTER than truncated_sampling!!
        # ...plus solves the 'np.inf problem' when using "truncated_sampling".
        # S_DUR = truncated_sampling(AVGDUR[nreg], limits=dur_lim, n=NUM_S)

        # computing time
        MATE, STRIDE = quantum_time(DOYEAR[nreg], DATIME[nreg], DUR_S, date_pool, NUM_S)
        # plt.plot(range(len(MATE)), MATE, color='g')
        # pd.DataFrame({'time':MATE}).groupby('time').size().plot()
        group_idx = np.array(list(map(len, STRIDE))).cumsum()[:-1]
        time_idx = pd.DataFrame(MATE).groupby([0], sort=False).indices  # unsorted
        # time_idx = pd.DataFrame(MATE).groupby([0], sort=True).indices  # sorted

        # multiply and displace storm.centres
        M_CENT = moving_storm(DIRMOV[nreg], VELMOV[nreg], STRIDE, CENT.samples)
        # # run the line below instead for positive.speeds & constant.speed storms
        # M_CENT = moving_storm(DIRMOV[nreg], VELMOV[nreg], STRIDE, CENT.samples,
        #                       speed_lim=(0.01, np.inf), speed_stat='mean')
        M_CENT = np.array(reduce(iconcat, M_CENT, []))
        # https://stackoverflow.com/a/45323085/5885810  (list-of-lists flat)

        # sampling maxima radii
        RADII = faster_sampling(RADIUS[nreg], limits=(minmax_radius, np.inf), n=MATE.size)
        # RADII = np.split(RADII, group_idx)  # no.need.for.splitting.no.more?
        # 876 μs ± 4.81 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        # # run the line below instead for averaged/unique radii per storm
        # RADII = np.array(reduce(iconcat, list(map(
        #     lambda x: np.repeat(x.mean(), len(x)), np.split(RADII, group_idx))), []))

        # polygon(s) for maximum radii
        # ringo = list(map(last_ring, RADII, M_CENT))
        ringo = last_ring(RADII, M_CENT)
        # 1.48 ms ± 3.92 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
        # 29 ms ± 225 μs per loop (mean ± std. dev. of 7 runs, 10 loops each) [slower approach]

        if TACTIC == 1:
            # sampling max.intensity
            MAX_I = faster_sampling(MAXINT[nreg], limits=(NO_RAIN, upd_max), n=MATE.size)
            # MAX_I = np.split(MAX_I, group_idx)

            # sampling betas
            BETA = BETPAR[nreg][''].rvs(size=MATE.size)
            # BETA = faster_sampling(BETPAR[nreg], limits=(-np.inf, np.inf), n=MATE.size)
        else:
            # storm-center Z-tratification
            # https://stackoverflow.com/a/7015366/5885810  (list.map with 2 outputs)
            dem_ = abspath(join(parent_d, DEM_FILE))
            qant, ztat = elevation.retrieve_z(ringo, dem_, zones=Z_CUTS)
            # qant, ztat = elevation.retrieve_z(ringo.iloc[-1], dem_, zones=Z_CUTS)

            # compute copulas given the Z_bands (or not)
            _one, _two = zip(*qant.apply(lambda x: copula_sampling(
                COPONE[nreg], MAXINT[nreg], INTRAT[nreg], band=x['E'],
                n=x[Z_STAT]), axis='columns',))

            # concatenating & shuffling
            i_max = pd.concat(_one).values[ztat['in_id'].argsort()]
            I_RAT = pd.concat(_two).values[ztat['in_id'].argsort()]

            # pre-pairing for betas
            d_frame = pd.DataFrame({
                'pf_maxrainrate': i_max, 'rratio': I_RAT, 'radii': RADII,
                })
            # capping maxima above design.rain (or even max.possible.float)
            d_frame.loc[d_frame['pf_maxrainrate'] > upd_max, 'pf_maxrainrate'] = max_lim

            beto = betas(d_frame, method=None, seed=0.11, flag=False,
                         t_res=T_RES * TIME_DICT_['minutes'])
            # plt.hist(beto.df.loc[beto.df.flag!=1, 'beta'], bins=51, color='g')
            # plt.hist(beto.df['beta'], bins=51)
            BETA = beto.df['beta'].values

            # max.intensity as list.of.numpys
            MAX_I = beto.df['pf_maxrainrate'].values

        # # upgrading MAX_I according to STORMINESS (but this should be no more!!)
        # # using '(simy + 1)' starts the increase right from the first year
        # MAX_I = MAX_I * (1 + STORMINESS_SC[eval(n_sim_y)] + (simy * STORMINESS_SF[eval(n_sim_y)]))

        # compute granular rainfall over intermediate rings
        rings = list(map(lambda r, d, i, l, c: lotr(r, d, i, l, c),
                         RADII, BETA, MAX_I, reduce(iconcat, STRIDE, []), M_CENT))

        # group rings by unique time.stamp (to reduce # of rasterizations)
        c_ring, last_r = zip(*[
            [pd.concat([rings[x] for x in idx], ignore_index=True),
             pd.concat([ringo.iloc[x] for x in idx], ignore_index=True)] for\
                idx in list(time_idx.values())])
        # last_r = [pd.concat([ringo.iloc[x] for x in idx], ignore_index=True)
        #           for idx in list(time_idx.values())]
        # c_ring = list(map(lambda x: pd.concat(
        #     [rings[x] for x in x], ignore_index=True), list(time_idx.values())))
        # # # https://stackoverflow.com/a/38679861/5885810  (itemgetter)
        # # # ... but it did NOT allow for pd.concat (when only having one)
        # # c_ring = list(map(lambda x: pd.concat(itemgetter(*x)(rings), ignore_index=True), ...))
#%%
        # returns a time-sorted & void.trimmed (xarray) rainfall cube
        # the minimum value in the cube is the data.resolution (i.e., NO ZEROS)
        rain, suma = rain_cube(c_ring, last_r, list(time_idx.keys()), np_mask)
#%%
        # aggregate RAIN over iterations
        lrain.append(rain)
        lsuma.append(suma)

        #     guess = [1, len(suma)]
        #     gdiff = abs(np.diff(guess))[0]
        #     ii = 1
        #     while gdiff > 1:
        #         print(ii)
        #         # use_sum = suma[:guess[-1]]
        #         tmp_cum = xr.concat((lsuma[0], suma[:guess[-1]]), dim='time').groupby('time').sum().cumsum()
        #         pos_ = -1 if tmp_cum[-1] > train else +1
        #         guess[-1] = guess[-1] + pos_ * int(gdiff / 2)
        #         # guess = guess[::-1]  # reverse vector
        #         guess.reverse()
        #         gdiff = abs(np.diff(guess))[0]
        #         ii += 1
        #     fpos = min(guess)

        tmp_cum = suma.cumsum() + CUM_S  # tmp_cum.plot(color='g')
        # plt.plot(range(len(tmp_cum)), tmp_cum.time.data)

        # new_sum = xr.concat(lsuma, dim='time').groupby('time').sum() if\
        #     contar_int != 0 else lsuma[-1]
        # # plt.plot(range(len(new_sum)), new_sum.time.data)
        # tmp_cum = new_sum.cumsum()  # tmp_cum.plot(color='g')

        # preserve time.slices falling below "train"
        bynd = tmp_cum[tmp_cum >= train]['time'].data
        # update 'bynd' (in case 'train' hasn't been reached yet)
        bynd = bynd if len(bynd) == 0 else bynd[0].reshape(1)
        undr = np.concat([tmp_cum[tmp_cum < train]['time'].data, bynd],)
        # update time.step sums
        new_sum = xr.concat(
            (lsuma[0], lsuma[-1].loc[undr]), dim='time'
            ).groupby('time').sum() if contar_int != 0 else lsuma[-1].loc[undr]

        # idxs = np.concat([tmp_cum[tmp_cum < train]['time'].data, bynd],)

        # intersections
        # t_two = lsuma[-1]['time'].loc[np.isin(lsuma[-1]['time'], idxs)]
        t_two = lsuma[-1]['time'].loc[undr]
        # t_one = lsuma[0]['time'].loc[np.isin(lsuma[0]['time'], idxs)] if\
        #     contar_int != 0 else t_two
        t_one = lsuma[0]['time'] if contar_int != 0 else t_two
        intersect, one_ix, two_ix = np.intersect1d(
            t_one, t_two, assume_unique=True, return_indices=True)
        # intunsort = t_two[np.sort(two_ix)].data

        # DON'T update 'lsuma' before 'the intersections'
        # lsuma[0] = new_sum.loc[idxs]
        lsuma[0] = new_sum
        # lsuma[0] = new_sum.loc[intersect]

        if contar_int > 0:
            # painful but necessary ('u4') because maxima is clipped to 'zero'
            # 1 has to be subtracted 4.all intersected-sums after 1st iteration!

            # # # WHY NOT?? (saves one computation!)
            # # tmp_rain = xr.concat(list(map(lambda x: x.loc[intersect], lrain)),
            # #                      dim='time',).groupby('time').sum().astype('u4')
            # tmp_rain = xr.concat(
            #     list(map(lambda x: x.loc[np.isin(x['time'], idxs)], lrain)),
            #     dim='time',).groupby('time').sum().astype('u4')
            # # tmp_rain = xr.concat([
            # #     lrain[0].loc[np.isin(lrain[0]['time'], idxs)],
            # #     lrain[1].loc[np.isin(lrain[1]['time'], idxs)] - 1],
            # #     dim='time',).groupby('time').sum().astype('u4')
            tmp_rain = xr.concat((lrain[0], lrain[-1].loc[undr]), dim='time',
                                 ).groupby('time').sum().astype('u4')

            # # THIS IS also CORRECT!
            # tmp_rain.loc[intersect][tmp_rain.loc[intersect] >= 4] =\
            #     tmp_rain.loc[intersect][tmp_rain.loc[intersect] >= 4] - 1

            # '- 1' because summing.integers requires so (given ADD and SCL)
            # e.g., ((.004 - -.002) / .002) == 3; (3 * .002) - .002 == 0.004
            # the minima in both (left & right) is 2 (apart from zero)...
            # so everything above 4 should be subtracted 1, so the float is OK
            tmp_rain.loc[intersect] = tmp_rain.loc[intersect].where(
                tmp_rain.loc[intersect] < 4, tmp_rain.loc[intersect] - 1)
            # capping maxima again
            tmp_rain = tmp_rain.where(tmp_rain <= maxima, maxima)
        else:
            # tmp_rain = lrain[-1].loc[intersect]
            # np.equal(undr, intunsort).all()
            # tmp_rain = lrain[-1].loc[intunsort]
            tmp_rain = lrain[-1].loc[undr]
            # no need to cap maxima here (no sum of nothing here)
        # update also lrain[0] (so always to only have a 2-element list!)
        lrain[0] = tmp_rain.astype(RAINFMT)

        # remove last item so always to only have a 2-element list!
        del lrain[-1]
        del lsuma[-1]

        # update iterables
        kum_s = lsuma[0].cumsum()
        # CUM_S = tmp_cum.loc[idxs[-1]].data  # perhaps this is faster but dubious
        CUM_S = kum_s[-1].data
        NUM_S = int(NUM_S * .9)  # decrease the counter
        contar_int += 1

        # # output checking
        # print(f'\n CUMsum - PTOT: {CUM_S - train}\n----')
    # WARN IF THERE IS NO CONVERGING
    assert not (CUM_S < train and NUM_S < 2), f'Iteration not converging for '\
        f'REGION [{nreg}] -> YEAR [{simy}] !!\nTry a larger initial seed '\
        '(i.e., parameter "NUM_S"). If the problem persists, it might be '\
        'likely that the parameterization is not adequate.'

    collect()
    # lrain[0] = lrain[0].where(np_mask == 1, 0)
    # MAYBE THE 1.MASK SHOULD HAPPEN HERE
    return lrain[0], kum_s


# %% wrapper

def wrapper(NC_NAMES, year_z):
#%%
    global SPACE

    # set globals for INTEGER rainfall-NC-output
    nc_bytes()

    upd_max = np.min((MAXD_RAIN, iMAX)) if capmax_or_not == 1 else iMAX
    # upd_max = np.nanmin(np.array([MAXD_RAIN, iMAX], dtype='f4')) if\
    #     capmax_or_not == 1 else iMAX  # no need for NP.NANMIN; + is slower
    maxima = np.array(((upd_max - ADD) / SCL) + MINIMUM, dtype=RAINFMT)
    # transform to INTEGER the global rainfall maxima

    PDFS = read_pdfs()  # reads (and checks) the PDF-parameters
    construct_pdfs(PDFS)
    # pdfx = read_pdfs('./model_input/ProbabilityDensityFunctions_OND_3r.csv')
    # construct_pdfs(pdfx, tactic=2)

    SPACE = masking()  # SPACE.plot()  # SPACE = masking(catchment=SHP_FILE)

    region_s = regionalisation(
        # ZON_FILE.replace('.shp', f'_{SEASON_TAG}_{NREGIONS}kc.shp'),
        ZON_FILE.replace('.shp', f'_{SEASON_TAG}_{NREGIONS}r.shp'),
        'region', 'u_rain', SPACE,  # there is NO 'u_rain_ in KC.SHP
        )
    # plt.imshow(region_s['kmeans'], interpolation='none', cmap='turbo')
    # plt.imshow(region_s['npma'][-1], interpolation='none', cmap='plasma_r')

    if TER_FILE:
        icpac_s = regionalisation(TER_FILE, 'region', 'tercile', SPACE, add=-2)
        # plt.imshow(icpac_s['kmeans'], interpolation='none', cmap='turbo')
        # plt.imshow(icpac_s['npma'][-1], interpolation='none', cmap='plasma_r')
    else:
        # masks 1s to make.it icpac.compatible
        icpac_s = {'npma': [region_s['npma'][0].copy()]}
        icpac_s['npma'][0][:] = 1


#%%
    # FOR EVERY FILE/SIMULATION
    for nsim, sim_file in enumerate(NC_NAMES):
    # nsim=0; sim_file=NC_NAMES[nsim]

        print(f'\tRUN: {"{:02d}".format(nsim + 1)}/{"{:02d}".format(len(NC_NAMES))}')
        print('progress')
        print('********')

        nc = nc4.Dataset(sim_file, 'w', engine='h5netcdf')#, format='NETCDF4',)#set_auto_mask=False)
        nc.created_on = datetime.now(tzlocal()).strftime('%Y-%m-%d %H:%M:%S %Z')#%z

        # # 1ST FILL OF THE NC.FILE (defining global vars & CRS)
        # sub_grp, tag_y, tag_x = nc_file_i(nc, nsim,)
        nc, tag_y, tag_x = nc_file_iv(nc,)
        nc['regions'][:] = region_s['kmeans'].astype('i1')

        nc.close()

#%%
        # FOR EVERY YEAR of the SIMULATION
        for simy in tqdm(range(NUMSIMYRS), ncols=50):  # simy=0; year_z=2024

            iyear = year_z + simy
            mlen, date_pool = wet_days(iyear)
            # seasonal time.index
            time_seas = ((
                pd.date_range(date_pool[0], date_pool[1], freq=f'{T_RES}min',
                              inclusive='left', tz=TIME_ZONE) - date_origen
                ).total_seconds() * TIME_DICT_[TIME_OUTNC]).astype(TIMEINT).values

            # # 2ND FILL OF THE NC.FILE (creating the TIME & RAIN vars)
            nc = nc4.Dataset(sim_file, 'a', engine='h5netcdf')#, format='NETCDF4',)#set_auto_mask=False)

            sub_grp = nc_file_v(nc, iyear, time_seas, tag_y, tag_x,)
            sub_grp[RAIN_NAME].set_auto_mask(False)  # CRUCIAL for SPEED

            # sampling/updating total seasonal rainfall

            # # delete!!!
            # if ptot_or_kmean == 1:
            #     region_s['rain'] = pd.Series(np.ravel(list(map(
            #         lambda x: truncated_sampling(TOTALP[x], limits=(NO_RAIN, np.inf)),
            #         range(len(TOTALP))))), name='s_rain',)

            if ptot_or_kmean == 1:
                if TER_FILE:
                    # the lower.lim of 1st.element could lead to ZERO.rainfall!!
                    terlim = [[0., 1/3], [1/3, 2/3], [2/3, 1.]]
                    # https://numpy.org/doc/stable/reference/random/bit_generators/index.html
                    rtg = npr.Generator(npr.PCG64())
                    tercil = icpac_s['rain'].apply(lambda x: np.array(list(map(float, x.split('_')))))
                    tercil = tercil.apply(forecasting.split_diff)
                    wat = tercil.apply(lambda x: rtg.choice(3, size=1) if np.isnan(x).all() else rtg.choice(3, size=1, p=x / 100))
                    seas_rain = [[truncated_sampling(i, limits=terlim[j[0]], type_l='prob',) for j in wat] for i in TOTALP]
                else:
                    seas_rain = [[truncated_sampling(i, limits=(NO_RAIN, np.inf),)] for i in TOTALP]
            else:
                seas_rain = [[x] for x in list(map(np.asarray, region_s['rain'].tolist()))]

            C_OUT = []  # meaningless array to collect reached cums

            # FOR EVERY N_REGION
            for nreg, srain in enumerate(tqdm(seas_rain, ncols=50)):
            # nreg=2; srain=seas_rain[nreg]
                # print(nreg, srain)
                for jter, ireg in enumerate(tqdm(icpac_s['npma'], ncols=50)):
                # jter=1; ireg=icpac_s['npma'][jter]
                    micro_mask = region_s['npma'][nreg] * ireg
                    # plt.imshow(micro_mask, cmap='turbo', interpolation='none')
                # the region must be "large" enough to compute rainfall
                    if micro_mask.sum() > 999:

                        # scale (or not) the total seasonal rainfall
                        # using '(simy + 1)' starts the increase right from the first year
                        reg_tot = srain[jter] *\
                            (1 + PTOT_SC[eval(n_sim_y)] + (simy * PTOT_SF[eval(n_sim_y)]))
                            # (1 + PTOT_SC[simy] + (simy * PTOT_SF[simy]))
                        # reg_tot = 10.  # for testing!

                        reg_rain, cum_out = loop(
                            reg_tot, region_s['mask'].iloc[nreg], micro_mask,
                            nsim, simy, nreg, mlen, upd_max, maxima, date_pool,
                            )

                        # # where the rain must be placed
                        # what = np.intersect1d(time_seas, reg_rain['time'],
                        #                       assume_unique=True, return_indices=True)
                        # sub_grp[RAIN_NAME][what[1], :, :] = reg_rain.data +\
                        #     sub_grp[RAIN_NAME][what[1], :, :].astype(RAINFMT)
                        # # sub_grp[RAIN_NAME][sub_grp[RAIN_NAME] == 0] = 1

                        reg_rain = reg_rain.reindex({'time': time_seas}, fill_value=0)
                        sub_grp[RAIN_NAME][:] = reg_rain + sub_grp[RAIN_NAME][:].astype(RAINFMT)
                        # # having assigned the 'rain' name
                        # reg_rain.to_netcdf('./model_output/zdos.nc', engine='h5netcdf',
                        #     encoding={'rain':{'dtype':'u2', 'zlib':True, 'complevel':9}},
                        #     # encoding={'rain':{'dtype':'u2', 'compression':'gzip', "compression_opts": 9}},
                        #     )

                        collect()
                        # the line below should be removed??
                        C_OUT.append(cum_out[-1].data)

            # # FOR EVERY N_REGION
            # for nreg in tqdm(range(NREGIONS), ncols=50):  # nreg=2
            # # for nreg in tqdm(range(1), ncols=50):  # nreg=0  # for testing!

            #     # scale (or not) the total seasonal rainfall
            #     # using '(simy + 1)' starts the increase right from the first year
            #     reg_tot = region_s['rain'].iloc[nreg] *\
            #         (1 + PTOT_SC[eval(n_sim_y)] + (simy * PTOT_SF[eval(n_sim_y)]))
            #         # (1 + PTOT_SC[simy] + (simy * PTOT_SF[simy]))
            #     # reg_tot = 10.  # for testing!

            #     reg_rain, cum_out = loop(
            #         reg_tot, region_s['mask'].iloc[nreg], region_s['npma'][nreg],
            #         nsim, simy, nreg, mlen, upd_max, maxima, date_pool,
            #         )

            #     # # where the rain must be placed
            #     # what = np.intersect1d(time_seas, reg_rain['time'],
            #     #                       assume_unique=True, return_indices=True)
            #     # sub_grp[RAIN_NAME][what[1], :, :] = reg_rain.data +\
            #     #     sub_grp[RAIN_NAME][what[1], :, :].astype(RAINFMT)
            #     # # sub_grp[RAIN_NAME][sub_grp[RAIN_NAME] == 0] = 1

            #     reg_rain = reg_rain.reindex({'time': time_seas}, fill_value=0)
            #     sub_grp[RAIN_NAME][:] = reg_rain + sub_grp[RAIN_NAME][:].astype(RAINFMT)
            #     # # having assigned the 'rain' name
            #     # reg_rain.to_netcdf('./model_output/zdos.nc', engine='h5netcdf',
            #     #     encoding={'rain':{'dtype':'u2', 'zlib':True, 'complevel':9}},
            #     #     # encoding={'rain':{'dtype':'u2', 'compression':'gzip', "compression_opts": 9}},
            #     #     )

            #     collect()

            #     C_OUT.append(cum_out[-1].data)

            # MAYBE THE 1.MASK SHOULD HAPPEN HERE

            """
            this is the only right place to assign the SCL and ADD attributes\
            to the 'sub_grp[RAIN_NAME]' variable. ONLY HERE, and only after the\
            the whole variable has been set and filled up.
            you'd have serious problems (i.e., errors in the variable.values\
            when reading them (back) from the nc.file). please DO NOT be stupid!
            """
            if RAINFMT[0] != 'f':
                sub_grp[RAIN_NAME].scale_factor = SCL
                sub_grp[RAIN_NAME].add_offset = ADD

            nc.close()
            collect()
#%%

            # store.mean.stats as CSV.file (less memory when using INT)
            if output_stats_ == 1:
                zumaz = xr.DataArray(sub_grp[RAIN_NAME][:])
                # zumaz = zumaz.where(zumaz >= 2, 0)  # if storing 1's (ones)
                zumaz = zumaz.sum(dim='dim_0').astype('u4')
                zumaz = zumaz * SCL + ADD
                zumaz = zumaz.where(zumaz >= 0., 0.).round(3)  # if storing 0's
                cum_nc = []
                for rr in region_s['npma']:
                    cum_nc.append(zumaz.data[rr.astype(bool)].sum() / rr.sum())
                pd.DataFrame({
                    'y': np.repeat(iyear, NREGIONS),
                    'k': range(NREGIONS),
                    'mean_in': region_s['rain'].round(4),
                    'mean_out': [x.round(4) for x in C_OUT],
                    'mean_nc': [x.round(4) for x in cum_nc],
                    }).to_csv(sim_file.replace('.nc', '_stats.csv'), sep=',',
                              mode='a', index=False)


# def whopper(NC_NAMOS):
#     global SPACE
#     nc_bytes()
#     PDFS = read_pdfs()
#     construct_pdfs(PDFS)
#     SPACE = masking()
#     print('\nRUN PROGRESS')
#     print('************')
#     for nsim, sim_file in enumerate(NC_NAMOS):
#         nc = nc4.Dataset(sim_file, 'w', format='NETCDF4')
#         nc.created_on = datetime.now(tzlocal()).strftime('%Y-%m-%d %H:%M:%S %Z')#%z
#         nc, tag_y, tag_x = nc_file_iv(nc,)
#         for simy in tqdm(range(3), ncols=50):
#             iyear = year_z + simy
#             mlen, date_pool = wet_days(iyear)
#             time_seas = ((
#                 pd.date_range(date_pool[0], date_pool[1], freq=f'{T_RES}min',
#                               inclusive='left', tz=TIME_ZONE) - date_origen
#                 ).total_seconds() * TIME_DICT_[TIME_OUTNC]).astype(TIMEINT).values
#             # sub_grp = nc_file_iii(nc, iyear, time_seas,)
#             sub_grp = nc_file_v(nc, iyear, time_seas, tag_y, tag_x,)
#         nc.close()


# %% run

if __name__ == '__main__':

    from checks_ import welcome
    willkommen = welcome()
    NC_NAMES = willkommen.ncs
    wrapper(NC_NAMES, SEED_YEAR)
    # NC_NAMES = ['./model_output/01_test_xx.nc']
    # whopper(['./model_output/xxx.nc'])
