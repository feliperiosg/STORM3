import warnings

# https://stackoverflow.com/a/9134842/5885810     (supress warning by message)
warnings.filterwarnings('ignore', message='You will likely lose important projection '\
                        'information when converting to a PROJ string from another format')
# WOS doesn't deal with "ecCodes"
warnings.filterwarnings('ignore', message='Failed to load cfgrib - most likely '\
                        'there is a problem accessing the ecCodes library.')
# because the "EPSG_CODE = 42106" is not a standard proj?
warnings.filterwarnings('ignore', message="GeoDataFrame's CRS is not "\
                        "representable in URN OGC format")
# https://github.com/slundberg/shap/issues/2909    (suppresing the one from libpysal 4.7.0)
warnings.filterwarnings('ignore', message=".*`Geometry` class will deprecated '\
                        'and removed in a future version of libpysal*")
# # https://github.com/slundberg/shap/issues/2909    (suppresing the one from numba 0.59.0)
# warnings.filterwarnings('ignore', message=".*The 'nopython' keyword.*")

# https://stackoverflow.com/a/248066/5885810
from os.path import abspath, basename, dirname, join
parent_d = dirname(__file__)    # otherwise, will append the path.of.the.tests
# parent_d = './'               # to be used in IPython

import numpy as np
import pandas as pd
# # https://stackoverflow.com/a/65562060/5885810  (ecCodes in WOS)
import xarray as xr
# from pyproj import CRS
import pyproj as pp
import netCDF4 as nc4
import geopandas as gpd
from scipy import stats
from numpy import random as npr
from statsmodels.distributions.copula.api import GaussianCopula
# # from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d

from osgeo import gdal
# https://gdal.org/api/python_gotchas.html#gotchas-that-are-by-design-or-per-history
# https://github.com/OSGeo/gdal/blob/master/NEWS.md#ogr-370---overview-of-changes
if gdal.__version__.__getitem__(0) == '3':# enable exceptions for GDAL<=4.0
    gdal.UseExceptions()
    # gdal.DontUseExceptions()
    # gdal.__version__ # wos_ '3.6.2' # linux_ '3.7.0'

from rasterio import fill
# from rasterio import crs as rcrs
from zoneinfo import ZoneInfo
from datetime import timedelta, timezone, datetime
from dateutil.tz import tzlocal
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import rioxarray as rio

import libpysal as ps
from pointpats import PoissonPointProcess, random, Window  # , PointPattern

import matplotlib.pyplot as plt
from chunking import CHUNK_3D
from parameters import *
# from realization import READ_REALIZATION, REGIONALISATION#, EMPTY_MAP
from pdfs_ import circular, field, masking


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
STORM [STOchastic Rainfall Model] produces realistic regional or watershed rainfall under various
climate scenarios based on empirical-stochastic selection of historical rainfall characteristics.

# Based on Manuel F. Rios Gaona et al.
# [ https://doi.org/10.5194/gmd-17-5387-2024 ]

version name: STORM3

Authors:
    Manuel F. Rios Gaona 2023
Date created : 2023/05/11
Last modified: 2024/09/02
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


#%% INPUT PARAMETERS

"""
SEASONS = 1             # Number of Seasons (per Run)
NUMSIMS = 2#1#2         # Number of runs per Season
NUMSIMYRS = 1#2         # Number of years per run (per Season)

# # PARAMETER = [ S1 ]
PTOT_SC       = [0.00]
PTOT_SF       = [ 0.0]
STORMINESS_SC = [-0.0]
STORMINESS_SF = [+0.0]

# PDF_FILE = './model_input/ProbabilityDensityFunctions_ONE--ANALOG.csv'      # output from 'pre_processing.py'
# PDF_FILE = './model_input/ProbabilityDensityFunctions_ONE--ANALOG-pmf.csv'  # output from 'pre_processing.py'
SHP_FILE = './model_input/HAD_basin.shp'                # catchment shape-file in WGS84
# DEM_FILE = './model_input/dem/WGdem_wgs84.tif'        # aoi raster-file (optional**)
# DEM_FILE = './model_input/dem/WGdem_26912.tif'        # aoi raster-file in local CRS (***)
DEM_FILE = None
OUT_PATH = './model_output'                             # output folder

# RAIN_MAP = '../CHIMES/3B-HHR.MS.MRG.3IMERG.20101010-S100000-E102959.0600.V06B.HDF5'     # no.CRS at all!
# RAIN_MAP = './realisation_MAM_crs-wrong.nc'           # no..interpretable CRS
RAIN_MAP = './realisation_MAM_crs-OK.nc'                # yes.interpretable CRS
SUBGROUP = ''
NREGIONS = 1#4#1                                        # number of regions to split the whole.region into

Z_CUTS = None               # (or Z_CUTS = []) for INT-DUR copula modelling regardless altitude
# Z_CUTS = [1350, 1500]     # in meters!
Z_STAT = 'mean'#'median'    # statistic to retrieve from the DEM ['median'|'mean' or 'min'|'max'?? not 'count']

# OGC-WKT for HAD [taken from https://epsg.io/42106]
WKT_OGC = 'PROJCS["WGS84_/_Lambert_Azim_Mozambique",'\
    'GEOGCS["unknown",'\
        'DATUM["unknown",'\
            'SPHEROID["Normal Sphere (r=6370997)",6370997,0]],'\
        'PRIMEM["Greenwich",0,'\
            'AUTHORITY["EPSG","8901"]],'\
        'UNIT["degree",0.0174532925199433,'\
            'AUTHORITY["EPSG","9122"]]],'\
    'PROJECTION["Lambert_Azimuthal_Equal_Area"],'\
    'PARAMETER["latitude_of_center",5],'\
    'PARAMETER["longitude_of_center",20],'\
    'PARAMETER["false_easting",0],'\
    'PARAMETER["false_northing",0],'\
    'UNIT["metre",1,'\
        'AUTHORITY["EPSG","9001"]],'\
    'AXIS["Easting",EAST],'\
    'AXIS["Northing",NORTH],'\
    'AUTHORITY["EPSG","42106"]]'
# # ESRI-WKT for HAD [alternative?]
# WKT_OGC = 'PROJCS["WGS84 / Lambert Azim Mozambique",'\
#     'GEOGCS["WGS 84",'\
#         'DATUM["WGS_1984",'\
#             'SPHEROID["WGS_1984",6378137,298.257223563]],'\
#         'PRIMEM["Greenwich",0],'\
#         'UNIT["Decimal_Degree",0.0174532925199433]],'\
#     'PROJECTION["Lambert_Azimuthal_Equal_Area"],'\
#     'PARAMETER["latitude_of_origin",5],'\
#     'PARAMETER["central_meridian",20],'\
#     'UNIT["Meter",1]]'
# # ---------------------------------------------------
# # OGC-WKT for WGS84 [taken from https://epsg.io/4326]
# WGS84_WKT = 'GEOGCS["WGS 84",'\
#     'DATUM["WGS_1984",'\
#         'SPHEROID["WGS 84",6378137,298.257223563,'\
#             'AUTHORITY["EPSG","7030"]],'\
#         'AUTHORITY["EPSG","6326"]],'\
#     'PRIMEM["Greenwich",0,'\
#         'AUTHORITY["EPSG","8901"]],'\
#     'UNIT["degree",0.0174532925199433,'\
#         'AUTHORITY["EPSG","9122"]],'\
#     'AUTHORITY["EPSG","4326"]]'

# #~STORM.WAY~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# # USE ANY OF THE FOLLOWING X_Y PAIRS, IF USING "SHP_REGION() -> [BUFFER.based 1ST]"
BUFFER    =  8000.                      # in meters! -> buffer distance (out of the HAD)
X_RES     =  5000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
Y_RES     =  5000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
# X_RES     = 10000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
# Y_RES     = 10000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
# X_RES     =  1000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
# Y_RES     =  1000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
# #~STORM.WAY~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

CLOSE_DIS =  0.15                       # in km -> small circle emulating the storm centre's point/dot
MINRADIUS =  max([X_RES, Y_RES]) /1e3
RINGS_DIS =  MINRADIUS *(2) +.1         # in km -> distance between (rainfall) rings; heavily dependant on X_Y_RES

T_RES   =  30                           # in minutes! -> TEMPORAL.RES of TIME.SLICES
NO_RAIN =  0.01                         # in mm -> minimum preceptible/measurable/meaningful RAIN in all AOI
MIN_DUR =  2                            # in minutes!
MAX_DUR =  60*24*5                      # in minutes! -> 5 days (in this case)
# # OR:
# MIN_DUR =  []                           # use 'void' arrays if you want NO.CONSTRAINT on storm-duration
# MAX_DUR =  []                           # ... in either (or both) MIN_/MAX_DUR parameters/constants
# designed MAXIMUM rainfall for T_RES!! (Twice of that of IMERG!) [note also that 120 << 131.068_iMAX]
MAXD_RAIN = 60 *2                       # in mm
DISPERSE_ = .2                          # factor to split MAXD_RAIN into

# SEED_YEAR  = None                       # for your SIM/VAL to start in the current year
SEED_YEAR = 2023                        # for your SIM/VAL to start in 2050
### bear in mind the 'SEASONS' variable!... (when toying with 'SEASONS_MONTHS')
# SEASONS_MONTHS = [[6,10], None]         # JUNE through OCTOBER (just ONE season)
# # OR:
SEASONS_MONTHS = [['mar','may'], ['oct','dec']]     # OCT[y0] through MAY[y1] (& JULY[y1] through SEP[y1])
# SEASONS_MONTHS = [[10,5], ['jul','sep']]            # OCT[y0] through MAY[y1] (& JULY[y1] through SEP[y1])
# SEASONS_MONTHS = [['may','sep'],[11,12]]            # MAY through SEP (& e.g., NOV trhough DEC)
TIME_ZONE      = 'Africa/Addis_Ababa'               # Local Time Zone (see links below for more names)
# # OR:
# TIME_ZONE    = 'UTC'
# DATE_ORIGIN    = '1950-01-01'                       # to store dates as INT
DATE_ORIGIN    = '1970-01-01'                       # to store dates as INT

### only touch this parameter if you really know what you're doing!!)
# RAINFMT = 'f4'
RAINFMT = 'u2'                          # 'u' for UNSIGNED.INT  ||  'i' for SIGNED.INT  ||  'f' for FLOAT
                                        # number of Bytes (1, 2, 4 or 8) to store the RAINFALL variable (into)
# SIGNINT = 'u'                           # 'u' for UNSIGNED.INT  ||  'i' for SIGNED.INT  ||  'f' for FLOAT
# INTEGER = 2                             # number of Bytes (1, 2, 4 or 8) to store the RAINFALL variable (into)
# # INTEGER = 4                             # number of Bytes (1, 2, 4 or 8) to store the RAINFALL variable (into)
PRECISION = 0.002                       # output precision
# TIME dimension
TIMEINT = 'u4'                          # format for integers in TIME dimension
TIMEFIL = +(2**( int(TIMEINT[-1]) *8 )) -1
TIME_OUTNC = 'minutes'                  # UNITS (since DATE_ORIGIN) for NC.TIME dim
# TIME_DICT_ = dict(seconds=60 ,minutes=1, hours=1/60, days=(60*24)**-1)
TIME_DICT_ = dict(seconds=1 ,minutes=1/60, hours=1/60**2, days=1/(60**2*24))
"""


# %% switches

ptot_or_kmean = 1  # 1 if seasonal.rain sampled; 0 if taken from shp.kmeans


# %% FUNCTIONS' DEFINITION

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- SET UP SPACE-TIME DOMAIN & UPDATE PARAMETERS ---------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ replace FILE.PARAMETERS with those read from the command line ~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def PAR_UPDATE(args):
    # https://stackoverflow.com/a/2083375/5885810  (exec global... weird)
    for x in list(vars(args).keys()):
        exec(f'globals()["{x}"] = args.{x}')
        # print([PTOT_SC, PTOT_SF])


# %% something xtra

def wet_days():
    """
    defines the days of the (wet)-season to sample from.\n
    Input: none.\n
    Output ->
    year_z : int; modified (or not) SEED_YEAR.
    nonths : int; number of months in the season.
    datespool : list; start & end 'datetime.datetime(s)' of the season.\n
    """
    # establish the SEASONAL-dict
    sdic = {'MAM': [3, 4, 5], 'JJAS': [6, 7, 8, 9], 'OND': [10, 11, 12]}
    # update SEED_YEAR
    year_z = SEED_YEAR if SEED_YEAR else datetime.now().year
    # monthly duration for season(al)-[tag] (12 months in a year)
    nonths = sdic[SEASON_TAG][-1] - sdic[SEASON_TAG][0] + 1
    # construct the date.times
    ini_date = datetime(year=year_z, month=sdic[SEASON_TAG][0], day=1)
    datespool = [ini_date, ini_date + relativedelta(months=nonths)]
    datespool_p = [datespool[0], datespool[-1] - relativedelta(seconds=1)]
    # extract Day(s)-Of-Year(s)
    DOY_POOL = list(map(lambda d: d.timetuple().tm_yday, datespool_p))
    return year_z, nonths, datespool


# def SHP_REGION():
#     """
#     X_RES     =     5000.                   # in meters! (pxl.resolution for the 'regular/local' CRS)
#     Y_RES     =     5000.                   # in meters! (pxl.resolution for the 'regular/local' CRS)
#     """
#     global llim, rlim, blim, tlim, XS, YS, CATCHMENT_MASK, BUFFRX, BUFFRX_MASK
# # read WG-catchment shapefile (assumed to be in WGS84)
#     wtrwgs = gpd.read_file( abspath( join(parent_d, SHP_FILE) ) )
# # transform it into EPSG:42106 & make the buffer    # this code does NOT work!
# # https://gis.stackexchange.com/a/328276/127894     (geo series into gpd)
#     # wtrshd = wtrwgs.to_crs( epsg=42106 )
#     wtrshd = wtrwgs.to_crs( crs = WKT_OGC )          # //epsg.io/42106.wkt
#     BUFFRX = gpd.GeoDataFrame(geometry=wtrshd.buffer( BUFFER ))#.to_crs(epsg=4326)
# # infering (and rounding) the limits of the buffer-zone
#     llim = np.floor( BUFFRX.bounds.minx[0] /X_RES ) *X_RES #+X_RES/2
#     rlim = np.ceil(  BUFFRX.bounds.maxx[0] /X_RES ) *X_RES #-X_RES/2
#     blim = np.floor( BUFFRX.bounds.miny[0] /Y_RES ) *Y_RES #+Y_RES/2
#     tlim = np.ceil(  BUFFRX.bounds.maxy[0] /Y_RES ) *Y_RES #-Y_RES/2

# #~IN.CASE.YOU.WANNA.XPORT.(OR.USE).THE.MASK+BUFFER.as.geoTIFF~~~~~~~~~~~~~~~~~~#
#     # # ACTIVATE if IN.TIFF
#     # tmp_file = 'tmp-raster_mask-buff.tif'
#     # tmp = gdal.Rasterize(tmp_file, BUFFRX.to_json(), format='GTiff'
#     # ACTIVATE if IN.MEMORY
#     tmp = gdal.Rasterize('', BUFFRX.to_json(), format='MEM'#, add=0
#         , xRes=X_RES, yRes=Y_RES, noData=0, burnValues=1, allTouched=True
#         , outputType=gdal.GDT_Int16, outputBounds=[llim, blim, rlim, tlim]
#         , targetAlignedPixels=True
#         # , targetAlignedPixels=False # (check: https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap)
#     # UPDATE needed for outputSRS [in WKT instead of PROJ4]
#         , outputSRS=pp.CRS.from_wkt(WKT_OGC).to_proj4()
#         # # , width=(abs(rlim-llim)/X_RES).astype('u2'), height=(abs(tlim-blim)/X_RES).astype('u2')
#         )
#     BUFFRX_MASK = tmp.ReadAsArray().astype('u1')
#     tmp = None

# #~BURN THE CATCHMENT SHP INTO RASTER (WITHOUT BUFFER EXTENSION)~~~~~~~~~~~~~~~~#
# # https://stackoverflow.com/a/47551616/5885810  (idx polygons intersect)
# # https://gdal.org/programs/gdal_rasterize.html
# # https://lists.osgeo.org/pipermail/gdal-dev/2009-March/019899.html (xport ASCII)
# # https://gis.stackexchange.com/a/373848/127894 (outputing NODATA)
# # https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap (targetAlignedPixels==True)
#     tmp = gdal.Rasterize('', wtrshd.to_json(), format='MEM', add=0
#         , xRes=X_RES, yRes=Y_RES, noData=0, burnValues=1, allTouched=True
#         , outputType=gdal.GDT_Int16, outputBounds=[llim, blim, rlim, tlim]
#         , targetAlignedPixels=True
#         # ,targetAlignedPixels=False # (check: https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap)
#     # UPDATE needed for outputSRS [in WKT instead of PROJ4]
#         , outputSRS=pp.CRS.from_wkt(WKT_OGC).to_proj4()
#         # # , width=(abs(rlim-llim)/X_RES).astype('u2'), height=(abs(tlim-blim)/X_RES).astype('u2')
#         )
#     CATCHMENT_MASK = tmp.ReadAsArray().astype('u1')
#     tmp = None           # flushing!

# # DEFINE THE COORDINATES OF THE XY.AXES
#     XS, YS = list(map( lambda a,b,c: np.arange(a +c/2, b +c/2, c),
#                       [llim,blim], [rlim,tlim], [X_RES,Y_RES] ))
# # flip YS??
#     YS = np.flipud( YS )      # -> important...so rasters are compatible with numpys


def regionalisation(file_zon, tag, xpace):
    """
    arrange into a dictionary shp-regions.\n
    Input ->
    *file_zon* : char; path to rain-regions shapefile.
    *tag* : char; geoPandas.GeoDataFrame column used as burning values.
    *xpace* : class; class where spatial variables are defined.\n
    Output -> dict; ...
    """
    reg_shp = gpd.read_file(abspath(join(parent_d, file_zon)))
    # transform it into EPSG:42106 & make the buffer
    # https://gis.stackexchange.com/a/328276/127894  (geo series into gpd)
    reg_shp = reg_shp.to_crs(crs=xpace.wkt_prj)  # //epsg.io/42106.wkt
    # k-means for free
    # k_means = region_to_numpy(reg_shp, xpace)
    k_means = masking.shapsterize(
        reg_shp.to_json(), xpace.x_res, Y_RES, -1, tag,
        [xpace.bbbox[i] for i in ['l', 'b', 'r', 't']],
        pp.CRS.from_wkt(xpace.wkt_prj).to_proj4(),
        )
    # plt.imshow(k_means, interpolation='none', cmap='turbo')  # testing
    # numpy masks (for all the unique values but -1)
    # ... 1st transform K-mean into 1s (because of the 0 K-mean);
    # ... then assign 0 everywhere else.
    reg_np = list(map(lambda x: xr.DataArray(k_means).where(
        k_means != x, 1).where(k_means == x, 0).data,
        np.setdiff1d(np.unique(k_means), -1)))
    # as long as the GeoDataFrame comes from "pdfs_", it'll always have 'u_rain'
    output = dict(zip(
        ('mask', 'rain', 'npma', 'kmeans'),
        (reg_shp, reg_shp['u_rain'], reg_np, k_means)
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


def construct_pdfs(pdframe):
    """
    sets up and fills the global key pdf-parameters for STORM to work.\n
    Input ->
    *pdframe* : pd.DataFrame; dataframe with tabulated pdf-parameters.\n
    Output: None.
    """
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
        } if TACTIC == 1 else {
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


# #~ TOY PDFS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# def ALT_CHECK_PDF():
#     global DATIME, DOYEAR, COPULA, TOTALP, RADIUS, BETPAR, MAXINT, AVGDUR, WINDIR, WSPEED

#     TOTALP = [{'':stats.gumbel_l(5.5116, 0.2262)}, {'':stats.norm(5.3629, 0.3167)}]
#     RADIUS = [{'':stats.johnsonsb(1.5187, 1.2696, -0.2789, 20.7977)}, {'':stats.gamma(4.3996, -0.475, 1.399)}]
#     BETPAR = [{'':stats.exponnorm(8.2872, 0.0178 ,0.01)}, {'':stats.burr(2.3512, 0.85, -0.0011, 0.0837)}]
#     MAXINT = [{'':stats.expon(0.1057 ,6.9955)}] *2
#     # [{'Z1': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x1cca438a850>,
#     #   'Z2': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x1cca43257c0>,
#     #   'Z3': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x1cca4395be0>},
#     #  {'Z1': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x1cca438ec10>,
#     #   'Z2': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x1cca4325f40>,
#     #   'Z3': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x1cca438e3d0>}]
#     AVGDUR = [{'':stats.geninvgauss(-0.089, 0.77, 2.8432, 82.0786)}] *2
#     # COPULA = [{'':-0.31622}, {'':-0.31622}]
#     # # [{'Z1': -0.2764573348234358, 'Z2': -0.31246435519305843, 'Z3': -0.44038940293798295},
#     # #  {'Z1': -0.2764573348234358, 'Z2': -0.31246435519305843, 'Z3': -0.44038940293798295}]
#     COPULA = [{'':-0.31622, 'Z1':-0.276457, 'Z2':-0.312464, 'Z3':-0.44}] *2
#     DATIME = [{'p': np.r_[0.2470, 0.3315, 0.4215], 'mus': np.r_[0.6893, 1.7034, 2.5756],
#                'kappas': np.r_[6.418, 3., 0.464]},
#               {'p': np.r_[1.], 'mus': np.r_[1.444], 'kappas': np.r_[1.0543]}]
#     DOYEAR = [{'p': np.r_[0.054, 0.089, 0.07, 0.087, 0.700], 'mus': np.r_[1.9, 0.228, 1.55, 1.172, 0.558] -np.pi /2,
#                'kappas': np.r_[105.32,  51.97,  87.19,  52.91, 6.82]},
#               {'p': np.r_[1.], 'mus': np.r_[1.6136], 'kappas': np.r_[3.9]}]
#     DOYEAR = DOYEAR[::-1]
#     # a KAPPA~=0 tranforms a VM into a UNIFORM.DISTRO from [-pi,pi]
#     # https://rstudio.github.io/tfprobability/reference/tfd_von_mises.html
#     WINDIR = [{'p': np.r_[1.], 'mus': np.r_[np.pi *.9], 'kappas': np.r_[0.00001]},
#               {'p': np.r_[1.], 'mus': np.r_[np.pi *.1], 'kappas': np.r_[0.00001]}]
# # until you parameterize WIN.DIRECTION use scypi's VONMISES [it's ~100x faster than VMM-package!]
#     WINDIR = [{'':stats.vonmises(.9 *np.pi, .00001)}, {'':stats.vonmises(.1 *np.pi, .00001)}]
#     WSPEED = [{'':stats.norm(7.55, 1.9)}] *2


# %% something else



#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RANDOM SMAPLING --------------------------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# #~ N-RANDOM SAMPLES FROM 'ANY' GIVEN PDF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# def RANDOM_SAMPLING( PDF, N ):
# # PDF: scipy distribution_infrastructure (constructed PDF)
# # N  : number of (desired) random samples
#     xample = PDF.rvs( size=N )
#     # # for reproducibility
#     # xample = PDF.rvs( size=N, random_state=npr.RandomState(npr.Philox(12345)) )
#     # xample = PDF.rvs( size=N, random_state=npr.RandomState(npr.PCG64DXSM(1337)) )
#     return xample


# def seasonal_rain_e(totalp_dis, band='', n=1):
#     """
#     samples total monsoonal rainfall (with potential climatic drivers).
#     also guarantees that no negative values are sampled!.\n
#     Input: ->
#     *totalp_dis* : dict; contains a scipy.stats (pdf) frozen infrastructure.
#     *kwargs ->
#     band : char; key (of the 'totalp_dis' dictionary) addressing the frozen pdf.
#     n : int; numbers of (random) samples.
#     Output -> np.array of floats with n-samples (of seasonal rainfall rainrate).
#     """
#     m = n
#     seas_rain = []
#     while m > 0:
#         pos_rain = totalp_dis[band].rvs(size=m)
#         # # reproducibility...
#         # seas_rain = totalp_dis[band].rvs(size=n, random_state=npr.RandomState(npr.Philox(12345)))
#         # seas_rain = totalp_dis[band].rvs(size=n, random_state=npr.RandomState(npr.PCG64DXSM(1337)))
#         pos_rain = pos_rain[pos_rain > NO_RAIN]
#         m = n - len(pos_rain)
#         seas_rain.append(pos_rain)
#     seas_rain = np.concatenate(seas_rain)
#     return seas_rain


def truncated_sampling(distro, **kwargs):
    """
    sampling (truncated or not) preserving the n-requested.\n
    Input ->
    *distro* : char; output path of nc-file.\n
    **kwargs ->
    limits : tuple; variable limits to smaple within.
    band : char; key (of the 'totalp_dis' dictionary) addressing the frozen pdf.
    n : int; numbers of (random) samples.\n
    Output -> np.array of floats with n-samples.
    """
    limits = kwargs.get('limits', (-np.inf, np.inf))
    band = kwargs.get('band', '')
    n = kwargs.get('n', 1)
    # set up useful range from limits
    ulims = list(map(distro[band].cdf, limits))
    # sample via a uniform.PPF
    sample = distro[band].ppf(npr.uniform(low=ulims[0], high=ulims[-1], size=n))
    # # reproducibility...
    # sample = distro[band].ppf(npr.RandomState(npr.SFC64(54321)).uniform(ulims[0], ulims[-1], n))
    # sample = distro[band].ppf(npr.RandomState(npr.PCG64(2001)).uniform(ulims[0], ulims[-1], n))
    # sample = distro[band].rvs(size=n, random_state=npr.RandomState(npr.Philox(12345)))
    # sample = distro[band].rvs(size=n, random_state=npr.RandomState(npr.PCG64DXSM(1337)))
    return sample


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


#~ SAMPLE FROM A COPULA & "CONDITIONAL" I_MAX-AVG_DUR PDFs ~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def COPULA_SAMPLING( COP, seas, BAND='', N=1 ):
# create the copula & sample from it
# https://stackoverflow.com/a/12575451/5885810  (1D numpy to 2D)
# (-1, 2) -> because 'GaussianCopula' will always give 2-cols (as in BI-variate copula)
# the 'reshape' allows for N=1 sampling
    IntDur = GaussianCopula(corr=COP[ seas ][ BAND ], k_dim=2).rvs( nobs=N ).reshape(-1, 2)
    # # for reproducibility
    # IntDur = GaussianCopula(corr=COP[ seas ][ BAND ], k_dim=2).rvs(
    #     nobs=N, random_state=npr.RandomState(npr.PCG64(20220608))).reshape(-1, 2)
    i_max = MAXINT[ seas ][ BAND ].ppf( IntDur[:, 0] )
    s_dur = AVGDUR[ seas ][ BAND ].ppf( IntDur[:, 1] )
    return i_max, s_dur


MATES, i_scaling = QUANTIZE_TIME( NUM_S, seas, simy, DUR_S[ okdur ] )



def base_round(stamps, **kwargs):
    """
    rounds time.stamps to either floor or nearest T_RES parameter.\n
    Input ->
    *stamps* : np.array; float numpy representing seconds since some origin.\n
    **kwargs ->
    base : int; rounding temporal resolution.
    time_tag : char; string indicating the base resolution.
    time_dic : dic; dictionary containing the equivalences of 'tags' in base 60.
    method : char; rounding method (either 'floor' or 'nearest').\n
    Output -> rounded numpy (to custom temporal resolution, i.e., T_RES).
    """
    base = kwargs.get('base', T_RES)
    time_dic = kwargs.get('time_dic', TIME_DICT_)
    time_tag = kwargs.get('time_tag', TIME_OUTNC)
    kase = kwargs.get('method', 'floor')
    # https://stackoverflow.com/a/2272174/5885810
    # update 'base'
    base = base * time_dic[time_tag] * 60  # the system/dic is base.60 (-> * 60)
    if kase == 'floor':
        iround = (base * (np.ceil(stamps / base) - 1))  # .astype(TIMEINT)
    elif kase == 'nearest':
        iround = (base * (stamps / base).round())  # .astype(TIMEINT)
    else:
        raise TypeError("Wrong method passed!\n"
                        "Pass 'floor' or 'nearest' to the 'method' argument.")
    return iround


#~ [TIME BLOCK] SAMPLE DATE.TIMES and XPAND THEM ACCORDING MOVING VECTORS ~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def QUANTIZE_TIME( NUM_S, seas, simy, durations ):# durations=DUR_S
    # global i_scaling
# sample some dates (to capture intra-seasonality & for NC.storing)
    DATES = TOD_CIRCULAR( NUM_S, seas, simy )
# round starting.dates to nearest.floor T_RES
    RATES = base_round( DATES )
# turn the DUR_S into discrete time.slices
    i_scaling = TIME_SLICE( (DATES+ -1*RATES), durations )
# xploding of discrete timestamps (per storm.cluster)
    MATES = np.concatenate( list(map(lambda r_s,i_s:
        # np.arange(start=r_s, stop=r_s + 60*T_RES*len(i_s), step=60*T_RES),
        np.arange(start=r_s, stop=r_s + (T_RES *TIME_DICT_[ TIME_OUTNC ] *60) *len(i_s),
                  step=T_RES *TIME_DICT_[ TIME_OUTNC ] *60),
        RATES, i_scaling)) ).astype( TIMEINT )
    return MATES, i_scaling


def xxx(doy_par, tod_par, n, simy, date_origen):
    # doy_par=DOYEAR[nreg]; tod_par=DATIME[nreg]; n=NUM_S

    # computing DOY
    M = n
    all_dates = []
    while M > 0:
        cs_day = circular(doy_par,)
        doys = cs_day.samples(M, data_type='doy')
        # cs_day.plot_samples(data=doys, data_type='doy', bins=40)  # plotting
        doys = doys[doys > 0]  # negative doys?? (do they belong to jan/dec?)
        # into actual dates
        dates = list(map(lambda d: datetime(year=DATE_POOL[0].year, month=1, day=1)
                         + relativedelta(yearday=int(d)), doys.round(0)))
        sates = pd.Series(dates)  # to pandas
        # chopping into limits
        sates = sates[(sates >= DATE_POOL[0]) & (sates <= DATE_POOL[-1])]
        M = len(dates) - len(sates)
        # updating to SIMY year (& storing)
        all_dates.append(sates.map(lambda d: d + relativedelta(years=simy)))
    all_dates = pd.concat(all_dates, ignore_index=True)

    # computing TOD
    cs_tod = circular(tod_par,)
    times = cs_tod.samples(n, data_type='tod')
    # cs_tod.plot_samples(data=times, data_type='tod', bins=40)  # plotting
# SECONDS since DATE_ORIGIN
# https://stackoverflow.com/a/50062101/5885810
    stamps = np.asarray(list(map(lambda d, t: (d + timedelta(hours=t) -
                                               date_origen).total_seconds(),
                                 all_dates.dt.tz_localize(TIME_ZONE), times)))
    # # https://stackoverflow.com/a/67105429/5885810  (chopping milliseconds)
    # stamps = np.asarray(
    #     list(map(lambda d, t: (d + timedelta(hours=t)).isoformat(
    #         timespec='seconds'), dates, times)))
    stamps = stamps * TIME_DICT_[TIME_OUTNC]  # scaled to output.TIMENC.res


# sample some dates (to capture intra-seasonality & for NC.storing)
    DATES = TOD_CIRCULAR( NUM_S, seas, simy )


    # round starting.dates to nearest.floor T_RES
    rates = base_round(stamps)

# turn the DUR_S into discrete time.slices
    i_scaling = TIME_SLICE( (DATES+ -1*rates), durations )
# xploding of discrete timestamps (per storm.cluster)
    MATES = np.concatenate( list(map(lambda r_s,i_s:
        # np.arange(start=r_s, stop=r_s + 60*T_RES*len(i_s), step=60*T_RES),
        np.arange(start=r_s, stop=r_s + (T_RES *TIME_DICT_[ TIME_OUTNC ] *60) *len(i_s),
                  step=T_RES *TIME_DICT_[ TIME_OUTNC ] *60),
        rates, i_scaling)) ).astype( TIMEINT )
    return MATES, i_scaling



#~ SAMPLE DAYS.OF.YEAR and TIMES.OF.DAY (CIRCULAR approach) ~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def TOD_CIRCULAR( N, seas, simy ):# N=120
    M = N
    all_dates = []
    while M>0:
        doys = vonmises.tools.generate_mixtures( p=DOYEAR[ seas ]['p'],
            mus=DOYEAR[ seas ]['mus'], kappas=DOYEAR[ seas ]['kappas'], sample_size=M)
# to DOY
        doys = (doys +np.pi) /(2*np.pi) *365 -1
    # negatives are giving me troubles (difficult to discern if they belong to january/december)
        doys = doys[ doys>0 ]
        # # to check out if the sampling is done correctly
        # plt.hist(doys, bins=365)
# into actual dates
        dates = list(map(lambda d:
            datetime(year=DATE_POOL[ seas ][0].year,month=1,day=1) +\
            relativedelta(yearday=int( d )), doys.round(0) ))
        sates = pd.Series( dates )              # to pandas
# chopping into limits
        sates = sates[(sates>=DATE_POOL[ seas ][0]) & (sates<=DATE_POOL[ seas ][-1])]
        M = len(dates) - len(sates)
        # print(M)
# updating to SIMY year (& storing)
        # all_dates.append( sates + pd.DateOffset(years=simy) )
        all_dates.append( sates.map(lambda d:d +relativedelta(years=simy)) )
        # # the line above DOES NOT give you errors when dealing with VOID arrays
    all_dates = pd.concat( all_dates, ignore_index=True )

    """
If you're doing "CIRCULAR" for DOY that means you did install "vonMisesMixtures"
... therefore, sampling for TOD 'must' also be circular (why don't ya)
    """
# TIMES
# sampling from MIXTURE.of.VON_MISES-FISHER.distribution
    times = vonmises.tools.generate_mixtures(p=DATIME[ seas ]['p'],
        mus=DATIME[ seas ]['mus'], kappas=DATIME[ seas ]['kappas'], sample_size=N)
# from radians to decimal HH.HHHH
    times = (times +np.pi) /(2*np.pi) *24
    # # to check out if the sampling is done correctly
    # plt.hist(times, bins=24)
# SECONDS since DATE_ORIGIN
# https://stackoverflow.com/a/50062101/5885810
    stamps = np.asarray( list(map(lambda d,t:
        (d + timedelta(hours=t) - DATE_ORIGEN).total_seconds(),
        all_dates.dt.tz_localize( TIME_ZONE ), times)) )
# # pasting and formatting
# # https://stackoverflow.com/a/67105429/5885810  (chopping milliseconds)
#     stamps = np.asarray(list(map(lambda d,t: (d + timedelta(hours=t)).isoformat(timespec='seconds'), dates, times)))
# STAMPS here are scaled to the output.TIMENC.resolution
    return stamps *TIME_DICT_[ TIME_OUTNC ]

    # # VISUALISATION
    # import matplotlib.pyplot as plt
    # # having 1st defined SEAS (down in "STORM")
    # PLTVAR = WINDIR
    # # PLTVAR = DOYEAR
    # number_of_samples = 200
    # samples = vonmises.tools.generate_mixtures( p=PLTVAR[ seas ]['p'],
    #     mus=PLTVAR[ seas ]['mus'], kappas=PLTVAR[ seas ]['kappas'], sample_size=number_of_samples)
    # # # 1.86 ms ± 6.32 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    # # # if using VONMISES from SCIPY
    # # loc = .9 * np.pi    # circular mean
    # # kappa = .00001      # concentration
    # # samples = stats.vonmises(loc=loc, kappa=kappa).rvs(number_of_samples)
    # # or
    # # samples = RANDOM_SAMPLING( WINDIR[ seas ][''], number_of_samples )
    # # # 19 µs ± 76.3 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)



#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RANDOM SMAPLING ----------------------------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RASTER MANIPULATION ----------------------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ CREATE AN OUTER RING/POLYGON ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# *1e3 to go from km to m
def LAST_RING( all_radii, CENTS ):# all_radii=RADII
# "resolution" is the number of segments in which a.quarter.of.a.circle is divided into.
# ...now it depends on the RADII/RES; the larger a circle is the more resolution it has.
    ring_last = list(map(lambda c,r: gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=[c[0]], y=[c[1]] ).buffer( r *1e3,
            # resolution=int((3 if r < 1 else 2)**np.ceil(r /2)) ),
            # resolution=np.ceil(r /MINRADIUS) +1 ), # or maybe... "+1"??
            resolution=np.ceil(r /MINRADIUS) +2 ), # or maybe... "+2"??
        crs=WKT_OGC), CENTS, all_radii))
    return ring_last


#~ CREATE CIRCULAR SHPs (RINGS & CIRCLE) & ASSING RAINFALL TO C.RINGS ~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def LOTR( RADII, MAX_I, DUR_S, BETAS, CENTS ):
    all_radii = list(map(lambda r:
        np.r_[np.arange(r, CLOSE_DIS, -RINGS_DIS), CLOSE_DIS], RADII))

    all_rain = list(map(lambda i,d,b,r: list(map( lambda r:
    # # model: FORCE_BRUTE -> a * np.exp(-2 * b * x**2)
    #     i * d *1/60 * np.exp( -2* b * r**2 ), r)), MAX_I, DUR_S, BETAS, all_radii))
    # model: BRUTE_FORCE -> a * np.exp(-2 * b**2 * x**2)
        i * d *1/60 * np.exp( -2* b**2 * r**2 ), r)), MAX_I, DUR_S, BETAS, all_radii))

# BUFFER_STRINGS
# https://www.knowledgehut.com/blog/programming/python-map-list-comprehension
# https://stackoverflow.com/a/30061049/5885810  (map nest)
# r,p are lists (at first instance), and the numbers/atoms (in the second lambda)
# .boundary gives the LINESTRING element
# *1e3 to go from km to m
# np.ceil(r /MINRADIUS) +2 ) is an artifact to lower the resolution of small circles
# ...a lower resolution in such circles increases the script.speed in the rasterisation process.
    rain_ring = list(map(lambda c,r,p: pd.concat( list(map(lambda r,p: gpd.GeoDataFrame(
        {'rain':p, 'geometry':gpd.points_from_xy( x=[c[0]], y=[c[1]] ).buffer( r *1e3,
            # resolution=int((3 if r < 1 else 2)**np.ceil(r /2)) ).boundary},
            # resolution=np.ceil(r /MINRADIUS) +1 ).boundary}, # or maybe... "+1"??
            resolution=np.ceil(r /MINRADIUS) +2 ).boundary}, # or maybe... "+2"??
        crs=WKT_OGC) , r, p)) ), CENTS, all_radii, all_rain))
# # the above approach (in theory) is much? faster than the list.comprehension below
#     rain_ring = [pd.concat( gpd.GeoDataFrame({'rain':p, 'geometry':gpd.points_from_xy(
#         x=[c[0]], y=[c[1]] ).buffer(r *1e3, np.ceil(r /MINRADIUS) +2 ).boundary},
#         crs=WKT_OGC) for p,r in zip(p,r) ) for c,r,p in zip(CENTS, all_radii, all_rain)]

    return rain_ring


#~ RASTERIZE SHPs & INTERPOLATE RAINFALL (between rings) ~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def RASTERIZE( ALL_RINGS ):# posx=23; ALL_RINGS=RINGS[posx]
# burn the ALL_RINGS
    tmp = gdal.Rasterize('', ALL_RINGS.to_json(), xRes=X_RES, yRes=Y_RES, allTouched=True,
        attribute='rain', noData=0, outputType=gdal.GDT_Float64, targetAlignedPixels=True,
        outputBounds=[llim, blim, rlim, tlim], outputSRS=pp.CRS.from_wkt(WKT_OGC).to_proj4(), format='MEM')
        #, width=int(abs(rlim-llim)/X_RES), height=int(abs(tlim-blim)/X_RES) )
    fall = tmp.ReadAsArray()
    tmp = None
    # gdal.Unlink('the_tmpfile.tif')
# burn the mask
# convert LINESTRING to POLYGON (in shapely). ".iloc[0]" for the largest/outter RADII
# https://stackoverflow.com/a/2975194/5885810
    OUTER_RING = ( ALL_RINGS.geometry.iloc[0] ).convex_hull
# create a GEOPANDAS from a SHAPELY so you can JSON.it
# https://stackoverflow.com/a/51520122/5885810
    tmp = gdal.Rasterize('',
        gpd.GeoSeries([ OUTER_RING ]).to_json(), xRes=X_RES, yRes=Y_RES, allTouched=True,
        burnValues=1, noData=0, outputType=gdal.GDT_Int16, targetAlignedPixels=True,
        outputBounds=[llim, blim, rlim, tlim], outputSRS=pp.CRS.from_wkt(WKT_OGC).to_proj4(), format='MEM')
        #, width=int(abs(rlim-llim)/X_RES), height=int(abs(tlim-blim)/X_RES) )
    mask = tmp.ReadAsArray()
    tmp = None
# re-touching the mask...to do a proper interpolation
    mask[np.where(fall!=0)] = 0
# everything that is 1 is interpolated
    fill.fillnodata(np.ma.array(fall, mask=mask), mask=None, max_search_distance=4.0, smoothing_iterations=2)
    return fall


#~ COMPUTE STATS OVER A DEM.RASTER (GIVEN A SHP.POLYGON) ~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def ZTRATIFICATION( Z_OUT ):
    # global qants, ztats
    if Z_CUTS:
# calculate zonal statistics
        # test = zonal_stats(abspath( join(parent_d, SHP_FILE) ),
        #     './data_WG/dem/WGdem_wgs84.tif', stats='count min mean max median')#??
        # IF YOUR DEM IS IN WGS84... RE-PROJECT THE POLYGONS TO 4326 (WGS84)
        ztats = zonal_stats(vectors=Z_OUT.to_crs(epsg=4326).geometry,
            raster=abspath( join(parent_d, DEM_FILE) ), stats=Z_STAT)
        # # OTHERWISE, A FASTER APPROACH IS HAVING THE DEM/RASTER IN THE LOCAL CRS
        # # ...i.e., DEM_FILE=='./data_WG/dem/WGdem_26912.tif'
        # ztats = zonal_stats(vectors=Z_OUT.geometry,
        #     raster=abspath( join(parent_d, DEM_FILE) ), stats=Z_STAT)
# to pandas
        ztats = pd.DataFrame( ztats )
# column 'E' classifies all Z's according to the CUTS
        ztats['E'] = pd.cut(ztats[ Z_STAT ], bins=cut_bin, labels=cut_lab, include_lowest=True)
        ztats.sort_values(by='E', inplace=True)
# storm centres/counts grouped by BAND
# https://stackoverflow.com/a/20461206/5885810  (index to column)
        qants = ztats.groupby(by='E').count().reset_index(level=0)
    else:
# https://stackoverflow.com/a/17840195/5885810  (1-row pandas)
        # qants = pd.DataFrame( {'E':'', 'median':len(Z_OUT)}, index=[0] )
        qants = pd.DataFrame( {'E':'', Z_STAT:len(Z_OUT)}, index=[0] )
        # ztats = pd.DataFrame( {'E':np.repeat('',len(Z_OUT))} )
        ztats = pd.Series( range(len(Z_OUT)) )      # 5x-FASTER! than the line above
    return qants, ztats

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RASTER MANIPULATION ------------------------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- MISCELLANEOUS TO TIME-DISCRETIZATION ------------------------------ (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



#~ SPLIT STORM.DURATION INTO DISCRETE/REGULAR TIME.SLICES ~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def TIME_SLICE( DIFF_ATES, S_DUR ):# DIFF_ATES=(DATES+ -1*RATES); S_DUR=DUR_S[ okdur ]
# how much time to the right (from STARTING.TIME) you have in the 1st slice
    LEFT_DUR = 1 -DIFF_ATES /(T_RES *TIME_DICT_[ TIME_OUTNC ] *60)
# how many complete-and-remaining tiles you have for slicing
    CENT_DUR = S_DUR /T_RES -LEFT_DUR   # S_DUR & T_RES always in minutes!
# negatives imply storm.duration smaller than slice [so update accordingly the 1st slice]
    LEFT_DUR[ CENT_DUR <0 ] = S_DUR[ CENT_DUR <0 ] /T_RES
# extract the number of complete-centered slices
    # CENT_INT = CENT_DUR.astype( 'i8' )
    CENT_INT = CENT_DUR.astype( TIMEINT )
# the remains of the complete-centered slices is what goes to the last slice [+ remove negatives]
    RIGH_DUR = CENT_DUR - CENT_INT
    RIGH_DUR[ RIGH_DUR < 0 ] = np.nan
# establish/repeat the number of centered-whole slices
# https://stackoverflow.com/a/3459131/5885810
    CENT_INT = list(map(lambda x,y:[x]*y, [1] * len(CENT_INT), CENT_INT))
# join LEFT, CENTER, and RIGHT slices
    sfactors = list(map(np.hstack, np.stack([LEFT_DUR,
        np.array(CENT_INT, dtype='object'), RIGH_DUR], axis=1).astype('object') ))
# remove NANs, and apply rainfall.scalability factor
# '/60' (60mins in 1h) because T_RES -> minutes & S_DUR -> mm/h [ALWAYS!]
    sfactors = list(map(lambda x:x[~np.isnan(x)] * T_RES /60, sfactors))
    return sfactors




#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- MISCELLANEOUS TO TIME-DISCRETIZATION -------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- EXTRA-CHUNK OF MISCELLANEOUS FUNCTIONS ---------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# #~ INDEX ALL DURATIONS OUTSIDE [MIN_DUR, MAX_DUR].range ~~~~~~~~~~~~~~~~~~~~~~~#
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# def CHOP( DUR_S ):
# # find the indexes outside the limits
#     outdur = np.concatenate((np.where(DUR_S<MIN_DUR).__getitem__(0) if MIN_DUR else np.empty(0),
#         np.where(DUR_S>MAX_DUR).__getitem__(0) if MAX_DUR else np.empty(0))).astype('int')
#     d_bool = ~np.in1d(range(len(DUR_S)), outdur)
# # update 'vectors'
#     return d_bool


#~ UPSCALE A LIST [GIVENG A VECTOR OF 'REPETITVE' VALUES/PATTERNS] ~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def XPAND( LIZT, NREP ):# NREP=i_scaling
    return list(map(lambda x:np.repeat(x, list(map(len, NREP))), LIZT))


#~ MOVE STORM CENTRES ALONG A WIND.DIR [RETURNING DISPLACED/MOVING CENTRES] ~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def MOVING_STORM( XENTS, i_scaling, seas ):# XENTS=CENTS[ okdur ]
# WIND.DIRECTION
# sampling from MIXTURE?? of VON_MISES-FISHER.distribution
    # angles = vonmises.tools.generate_mixtures(p=WINDIR[ seas ]['p'],
    #     mus=WINDIR[ seas ]['mus'], kappas=WINDIR[ seas ]['kappas'], sample_size=len(XENTS))
    angles = RANDOM_SAMPLING( WINDIR[ seas ][''], len(XENTS) )
# from radians to azimuth
    azimut = angles +np.pi/2
# WIND.SPEED
# sample wind speed [in km/h]
    wspeed = RANDOM_SAMPLING( WSPEED[ seas ][''], len(XENTS) )
# # no need for adjusting to T_RES as 'i_scaling' does so
#     wspeed = wspeed *T_RES /60  # because T_RES is in 'minutes'
# silly upscaling so one can see it's moving (COMMENT OUT in PRODUCTION!!)
    wspeed = wspeed *2.9

# DISPLACE THE STORM_CENTRES
# there might be VOIDs in 'i_scaling' as we don't need tha last.element
    stride = list(map( np.cumsum,
        # list(map(np.multiply, wspeed *1.9, list(map(lambda x:np.r_[0,x[:-1]], i_scaling)) )) ))
        list(map(np.multiply, wspeed, list(map(lambda x:np.r_[0,x[:-1]], i_scaling)) )) ))
    # deltax = list(map(np.multiply, stride *1000, np.cos(azimut)))
    # deltay = list(map(np.multiply, stride *1000, np.sin(azimut)))
    deltax = list(map(lambda s,a:s*1000 *np.cos(a), stride, azimut))
    deltay = list(map(lambda s,a:s*1000 *np.sin(a), stride, azimut))
# 'strides' are in km!
    x_s = list(map(np.add, XENTS[:,0], deltax))
    y_s = list(map(np.add, XENTS[:,1], deltay))
# put them in the same format as input
    n_cent = np.stack((np.concatenate(x_s),np.concatenate(y_s)), axis=1)
    return n_cent


#~ SORT A RAIN.CUBE [in Z-TIME] GIVEN SOME TIMESTAMPS ~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def SORT_CUBE( CUBO, DATO ):
# CUBO: a 3D-numpy unsorted in the Z-dim (consistent with DATO)
# DATO: an unsorted array of dates
    s_mate = np.argsort( DATO )
# SORT cube & AGGREGATE SLICES WITH IDENTICAL TIMESTAMPS
# https://stackoverflow.com/a/51297779/5885810
# https://stackoverflow.com/a/55354743/5885810  # split a numpy (by unique elements)
# U_MATE is the updated/chopped/sorted MATES
    u_mate, i_mate = np.unique(DATO[s_mate], return_index=True)
    # # we don't really need the counts here
    # u_mate, i_mate, c_mate = np.unique(DATO[s_mate], return_index=True, return_counts=True)
    cube = np.stack( list(map(lambda x:x.sum(axis=0),
        np.split(CUBO[s_mate,:,:], i_mate[1:], axis=0))), axis=0)
# "u_mate" & "cube" the updated DATO & CUBO (respectively)
    return cube, u_mate

    # """
    # aggreagating fields with equal timestamps migh be slower in NUMPY than in XARRAY;
    # therefore, groupingby (timestamps) and then summing may speed things up.
    # nevertheles as of 12/07/2023 this is not my believe, and has not been tested!!
    # """
    # # TOY.XAMPLE with XARRAYS & AGGRETAING THEM BY TIME.STAMPS
    # aa = np.array([[[0,0,0], [0,0,0], [0,0,0]], [[0,0,0], [1,1,0], [1,1,0]]])
    # aa = xr.DataArray(data=aa, dims=['time','row','col'])#, name='uno')
    # aa.coords['time'] = np.array([1685545200, 1685547000]).astype('u8')

    # bb = np.array([[[0,2,2], [0,2,2], [0,0,0]], [[0,0,0], [0,0,0], [0,0,0]]])
    # bb = xr.DataArray(data=bb, dims=['time','row','col'])#, name='uno')
    # bb.coords['time'] = np.array([1685547000, 1685548800]).astype('u8')

    # # cc = xr.merge([aa,bb], compat='equals', fill_value=0.)
    # cc = xr.concat([aa,bb], dim='time', join='inner')
    # cc = cc.groupby(group='time').sum()
    # # is.it.sorted? [apparently YES!]
    # cc.time.diff('time').min()
    # # cc.to_netcdf('cc.nc', mode='w')


    # # TOY.XAMPLE with MASKED.NUMPYS
    # uno = np.stack([np.full((3,3), i+1) for i in range(3)], axis=0)
    # mas = np.array([[1,1,1],[1,0,0],[1,0,0]])
    # # https://numpy.org/doc/stable/reference/generated/numpy.ma.array.html  # masked.numpy
    # # https://stackoverflow.com/a/37690587/5885810  # 2D mask.broadcasting into 3D
    # uno = np.ma.masked_array(uno, mask=np.broadcast_to(mas.astype('bool'), uno.shape))
    # # uno = MaskedArray(uno, mask=np.broadcast_to(mas.astype('bool'), uno.shape))
    # # https://numpy.org/doc/stable/reference/generated/numpy.apply_over_axes.html
    # dos = np.ma.apply_over_axes(np.sum, uno, [1,2]).data.ravel()
    # # np.diff(dos)


#~ SORTING THE.CUBE BY TIMESTAMPS [& AGGRETATING RAIN] & REMOVING VOID.FIELDS ~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def RAIN_CUBO( STORM_MATRIX, M_S, mate, NP_MASK ):# M_S=np.concatenate(i_scaling); mate=MATES
    tot_pix = NP_MASK.sum()             # pixels in mask
# squeeze the storms...
# here is where the 'proportionality'.multiplication happens
    cube = np.stack(STORM_MATRIX, axis=0) * M_S[:, None, None]
# sort the cube along the time.axis (+ it aggregates fields having the same.timestep)
    cube, mate = SORT_CUBE( cube, mate )

    '''
dividing by TOT/PIX here implies REACHING the MEAN (in the stopping criterium).
if the objective is REACHING the (granular) MEDIAN; something else has to be thought about!
    '''

# # IF appending NP_MASK...
# # https://numpy.org/doc/stable/reference/generated/numpy.ma.array.html  # masked.numpy
# # https://stackoverflow.com/a/37690587/5885810  # 2D mask.broadcasting into 3D
#     cube = np.ma.masked_array(cube, mask=np.broadcast_to(NP_MASK==0, cube.shape))
#         # mask=np.broadcast_to(~NP_MASK.astype('bool'), cube.shape))
# # aggregate over mask
#     suma = np.ma.apply_over_axes(np.sum, cube, [1,2]).data.ravel() /tot_pix

# IF multiplying NP_MASK...
    """
# HERE ALL RAIN.FALL.ING OUT OF THE MASK IS CHOPPED OUT!!
    """
    cube = cube *NP_MASK
    suma = np.apply_over_axes(np.sum, cube, [1,2]).ravel() /tot_pix

    cuma = np.cumsum( suma )

    # # to add void rainfall.fields
    # suma[3]=suma[4]; suma[-3]=suma[-4]; suma[0]=suma[1]; suma[-2]=suma[-1]

# WHAT TO REMOVE
# if rainfall within AOI is too little -> remove it too
    removez = np.where( suma < NO_RAIN /tot_pix ).__getitem__(0)
# if two consecutive aggretated fields are the "same" -> no storm fell within AOI
    removec = np.where( np.diff(cuma) < NO_RAIN /tot_pix ).__getitem__(0) +1
    dropone = np.union1d(removec, removez)

# remove VOID layers [located in DROPONE]
# https://stackoverflow.com/a/7429338/5885810
    cube = np.delete(cube, dropone, axis=0)
    mate = np.delete(mate, dropone, axis=0)
    suma = np.delete(suma, dropone, axis=0)

    # return cube.data, mate, suma    # if NP_MASK was appended
    return cube, mate, suma

    # # TESTING
    # test_cum = cube.sum(axis=0)   # after "cube, mate = SORT_CUBE( cube, mate )"
    # test_cum = np.ma.masked_array(test_cum, mask=NP_MASK==0)
    # # test_cum.sum() /tot_pix
    # # 1.6210998684945819

    # import matplotlib.pyplot as plt
    # plt.imshow(test_cum, origin='upper', vmin=.0, cmap='nipy_spectral_r')#, interpolation='nearest')
    # plt.imshow(NP_MASK, origin='upper', vmin=.0, cmap='nipy_spectral_r', interpolation='none')

    # # TOY.XAMPLE with MASKED.NUMPYS
    # uno = np.stack([np.full((3,3), i+1) for i in range(3)], axis=0)
    # mas = np.array([[1,1,1],[1,0,0],[1,0,0]])
    # # https://numpy.org/doc/stable/reference/generated/numpy.ma.array.html  # masked.numpy
    # # https://stackoverflow.com/a/37690587/5885810  # 2D mask.broadcasting into 3D
    # uno = np.ma.masked_array(uno, mask=np.broadcast_to(mas.astype('bool'), uno.shape))
    # # uno = MaskedArray(uno, mask=np.broadcast_to(mas.astype('bool'), uno.shape))
    # # https://numpy.org/doc/stable/reference/generated/numpy.apply_over_axes.html
    # dos = np.ma.apply_over_axes(np.sum, uno, [1,2]).data.ravel()
    # # np.diff(dos)

    # # ANOTHER VISUALISATION
    # # you have to run 'all_radii' [above in LOTR]
    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Circle

    # posx = 23#32#9
    # # first do 'fall' and/or 'mask' in RASTERIZE
    # da = xr.DataArray(data=fall, dims=['y','x'], coords={'x':XS, 'y':YS},)
    # pa = xr.DataArray(data=mask, dims=['y','x'], coords={'x':XS, 'y':YS},)
    # # FIND YOUR SLICES
    # da.isel( {'x':slice(115,130), 'y':slice(255,270)} ).plot(cmap='nipy_spectral_r', levels=10, vmin=5.5,vmax=15)
    # pa.isel( {'x':slice(115,130), 'y':slice(255,270)} ).plot(cmap='gist_ncar_r', levels=4, vmin=1,vmax=2)
    # # then one can plot the interpolated field (after doing STORM_MATRIX)
    # da = xr.DataArray(data=STORM_MATRIX[posx], dims=['y','x'], coords={'x':XS, 'y':YS},)

    # # TO PLOT IT RIGHT ONE HAS TO USE XARRAY!
    # fig, ax = plt.subplots(figsize=(9,7), dpi=200)
    # ax.set_aspect('equal')
    # # da.isel( {'x':slice(115,130), 'y':slice(200,215)} ).plot(
    # da.isel( {'x':slice(115,130), 'y':slice(255,270)} ).plot(
    #     cmap='nipy_spectral_r', levels=10, vmin=5.5,vmax=15, ax=ax)
    # # RINGS[23].plot(color='xkcd:bright aqua', lw=.47, ax=ax)
    # for rr in all_radii[posx] *1e3:
    #     circ = Circle((CENTS[posx][0], CENTS[posx][1]), rr, alpha=1, facecolor='None', lw=0.67,
    #         edgecolor=npr.choice(['xkcd:lime green','xkcd:gold','xkcd:electric pink','xkcd:azure']))
    #     ax.add_patch(circ)
    # # https://stackoverflow.com/a/64035939/5885810  (add vertical lines)
    # plt.vlines(x=np.arange(1920000,1990000,5000), ymin=-180000, ymax=-100000, colors='xkcd:off white', ls='dotted', lw=0.09)
    # plt.hlines(y=np.arange(-110000,-180000,-5000), xmin=1910000, xmax=1990000, colors='xkcd:off white', ls='dotted', lw=0.09)
    # # plt.show()
    # plt.savefig(f'tmp_raster_xarray_n{posx}_pol.pdf', bbox_inches='tight',pad_inches=0.02, facecolor=fig.get_facecolor())
    # plt.close()
    # plt.clf()

    # # mask
    # plt.imshow(CATCHMENT_MASK, interpolation='none', aspect='equal', origin='upper',
    #             cmap='gist_ncar_r', extent=(llim, rlim, blim, tlim))

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- EXTRA-CHUNK OF MISCELLANEOUS FUNCTIONS ------------------------------ (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# %% nc-file creation

def round_x(x, prec=3, base=PRECISION):
    """
    custom rounding to a specific base.\n
    Input ->
    *x* : float or np.array.\n
    **kwargs ->
    prec : int; significative digits.
    base : float; rounding precision (resolution).\n
    Output -> rounded float/np.array to the given precision/base.
    """
    # # https://stackoverflow.com/a/18666678/5885810
    return (base * (np.array(x) / base).round()).round(prec)


def nc_bytes():
    """
    scales 'down' floats to integers.\n
    Input: none.\n
    Output ->
    SCL : float; multiplicative scaling factor.
    ADD : float; additive scaling factor.
    MINIMUM : int; minimum integer allowed.\n
    """
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
    iMAX = PRECISION * (MAXIMUM - MINIMUM -1) + iMIN
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
# voll = round_x((allt * SCL) + ADD).round(4)
# (vall == voll).all()  # np.True_
    # if storing as FLOAT
    if RAINFMT[0] == 'f':
        SCL = 1.
        ADD = 0.
        MINIMUM = 0.
    return SCL, ADD, MINIMUM


def nc_file_i(nc, nsim, xpace, **kwargs):
    """
    skeleton of the (output) nc-file.\n
    Input ->
    *nc* : char; output path of nc-file.
    *nsim* : int; iterable of simulation.
    *xpace* : class; class where spatial variables are defined.\n
    **kwargs ->
    sref_name : char; name of the variable storing the CRS.\n
    Output ->
    sub_grp : nc.sub_group; nc variable storing the nsim-run.
    tag_y : char; coords-attribute in the Y-axis.
    tag_x : dict; coords-attribute in the X-axis.\n
    """
    sref_name = kwargs.get('sref_name', 'spatial_ref')

    # define SUB.GROUP and its dimensions
    sub_grp = nc.createGroup(f'run_{"{:02d}".format(nsim + 1)}')
    sub_grp.createDimension('y', len(xpace.ys))
    sub_grp.createDimension('x', len(xpace.xs))
    sub_grp.createDimension('n', NUMSIMYRS)

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
    grid = sub_grp.createVariable(sref_name, 'u1')
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
    yy = sub_grp.createVariable(
        'projection_y_coordinate', 'i4', dimensions=('y'),
        chunksizes=CHUNK_3D([len(xpace.ys)], valSize=4),
        )
    yy[:] = xpace.ys
    yy.coordinates = 'projection_y_coordinate'
    yy.long_name = 'y coordinate of projection'
    yy._CoordinateAxisType = 'GeoY'
    yy.grid_mapping = sref_name
    yy.units = 'meter'
    # storing local coordinates (X-axis)
    xx = sub_grp.createVariable(
        'projection_x_coordinate', 'i4', dimensions=('x'),
        chunksizes=CHUNK_3D([len(xpace.xs)], valSize=4),
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
    # grid = sub_grp.createVariable(sref_name, 'int')
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
    # yy = sub_grp.createVariable(
    #     'latitude', 'f8', dimensions=('y', 'x'),
    #     chunksizes=CHUNK_3D([len(xpace.ys), len(xpace.xs)], valSize=8),
    #     )
    # yy[:] = lat
    # yy.coordinates = 'latitude'
    # yy.long_name = 'latitude coordinate'
    # yy._CoordinateAxisType = 'Lat'
    # yy.grid_mapping = sref_name
    # yy.units = 'degrees_north'
    # # (X-axis)
    # xx = sub_grp.createVariable(
    #     'longitude', 'f8', dimensions=('y', 'x'),
    #     chunksizes=CHUNK_3D([len(xpace.ys), len(xpace.xs)], valSize=8),
    #     )
    # xx[:] = lon
    # xx.coordinates = 'longitude'
    # xx.long_name = 'longitude coordinate'
    # xx._CoordinateAxisType = 'Lon'
    # xx.grid_mapping = sref_name
    # xx.units = 'degrees_east'

    # store the MASK
    ncmask = sub_grp.createVariable(
        'mask', 'i1', dimensions=('y', 'x'), zlib=True, complevel=9,
        chunksizes=CHUNK_3D([len(xpace.ys), len(xpace.xs)], valSize=1),
        )
    ncmask[:] = xpace.catchment_mask
    ncmask.grid_mapping = sref_name
    ncmask.long_name = 'catchment mask'
    ncmask.description = '1 means catchment or region : 0 is void'
    ncmask.coordinates = f'{yy.getncattr("coordinates")} '\
        f'{xx.getncattr("coordinates")}'
    # # storing of some XTRA-VARIABLES:
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # e.g., "duration"...
    # ncxtra = sub_grp.createVariable(
    #     'duration', 'f4', dimensions=('t', 'n'),
    #     zlib=True, complevel=9, fill_value=np.nan,
    #     # fill_value=np.r_[0].astype('u2')),
    #     )
    # ncxtra.long_name = 'storm duration'
    # ncxtra.units = 'minutes'
    # ncxtra.precision = f'{1/60}'  # (1 sec); see last line of 'nc_file_ii'
    # ncxtra.grid_mapping = sref_name
    # # ncxtra.scale_factor = dur_SCL  # this would've to be estimated
    # # ncxtra.add_offset = dur_ADD  # this would've to be estimated
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # e.g., "sampled_total"...
    # iixtra = sub_grp.createVariable(
    #     'sampled_total', 'f4', dimensions=('n'),
    #     zlib=True, complevel=9, fill_value=np.nan,
    #     )
    # iixtra.long_name = 'seasonal total from PDF'
    # iixtra.units = 'mm'
    # iixtra.grid_mapping = sref_name
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # e.g., "k_means"...
    # # ... i don't think we should provide this variable!
    # maskxx = sub_grp.createVariable(
    #     'k_means', datatype='i1', dimensions=('y', 'x'),
    #     chunksizes=CHUNK_3D([len(xpace.ys), len(xpace.xs)], valSize=1),
    #     zlib=True, complevel=9, fill_value=np.array(-1).astype('i1'),
    #     # least_significant_digit=3
    #     )
    # maskxx.grid_mapping = sref_name
    # maskxx.long_name = 'k-means NREGIONS'
    # maskxx.description = '-1 indicates region out of any cluster'
    # maskxx.coordinates = f'{yy.getncattr("coordinates")} '\
    #     f'{xx.getncattr("coordinates")}'

    return sub_grp, yy.getncattr("coordinates"), xx.getncattr("coordinates")


def nc_file_ii(sub_grp, simy, yearz, dateo, ytag, xtag):
    """
    filling and closure of the (output) nc-file.\n
    Input ->
    *sub_grp* : nc.sub_group; nc variable storing the nsim-run.
    *simy* : int; iterable of the year of simulation.
    *yearz* : int; seeed year representing the y-simulation.
    *dateo* : datetime.datetime; DATE_ORIGIN + TIME_ZONE in datetime format.
    *tag_y* : char; coords-attribute in the Y-axis.
    *tag_x* : dict; coords-attribute in the X-axis.\n
    Output -> nc.sub_group; updated nc variable storing the nsim-run.\n
    """
    # define the TIME dimension (& variable)
    nctnam = f'time_{"{:03d}".format(simy + 1)}'  # for less than 1000 years
    sub_grp.createDimension(nctnam, None)
    timexx = sub_grp.createVariable(
        nctnam, TIMEINT, (nctnam), fill_value=TIMEFIL,
        )
    timexx.long_name = 'starting time'
    timexx.units = f"{TIME_OUTNC} since {dateo.strftime('%Y-%m-%d %H:%M:%S')}"
    # timexx.units = f"{TIME_OUTNC} since {dateo.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"
    timexx.calendar = 'proleptic_gregorian'  # 'gregorian'
    timexx._CoordinateAxisType = 'Time'
    timexx.coordinates = nctnam
    # define the RAINFALL variable
    ncvnam = f'year_{yearz + simy}'
    if RAINFMT[0] == 'f':
        # DOING.FLOATS
        ncvarx = sub_grp.createVariable(
            ncvnam, datatype=f'{RAINFMT}', dimensions=(nctnam, 'y', 'x'),
            zlib=True, complevel=9, least_significant_digit=3, fill_value=np.nan,
            )
    else:
        # DOING.INTEGERS
        ncvarx = sub_grp.createVariable(
            ncvnam, datatype=f'{RAINFMT}', dimensions=(nctnam, 'y', 'x'),
            zlib=True, complevel=9,
            fill_value=np.array(0).astype(f'{RAINFMT}'),  # 0 is filling!
            )
    ncvarx.precision = PRECISION
    ncvarx.long_name = 'rainfall'
    ncvarx.units = 'mm'
    ncvarx.grid_mapping = sub_grp['spatial_ref'].long_name
    ncvarx.coordinates = f'{ytag} {xtag}'
    # # define & fill some other XTRA variable (set up previously in "nc_file_i")
    # # ... XTRA, XTRAn, etc. should be passed to this function
    # # 'f4' guarantees 1-second (1/60 -minute) precision
    # sub_grp.variables['duration'][:, simy] = ((XTRA * 60).round(0) / 60).astype('f4')
    # # # https://stackoverflow.com/a/28425782/5885810  (round to the nearest-nth)
    # # sub_grp.variables['duration'][:, simy] =\
    # #     list(map(lambda x: round(x / (1 / 60)) * (1 / 60), XTRA))
    # sub_grp.variables['sampled_total'][simy] = XTRA2.astype('f4')
    return sub_grp


# %% core computation #1

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#-  CORE COMPUTATION & NC.FILE FILLING ------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ ZEROS INPLACE!! THE RAIN?.VALUES INSIDE A MASK ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def ZERO_FIELD( idxs, maxk ):
    ceros = ncvarx[ idxs, : ]
    # ceros = ceros * ~maxk.astype('bool')
    ceros = ceros * maxk
    ceros[ ceros==0 ] = MINIMUM
    ncvarx[ idxs, :] = ceros


#~ RETRIEVES IDXS OF NON-SURPASSING PXLS WITHIN THE MASK ~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def AUX_MSK( rain, sur_, NP_MASK, l_idx ):# l_idx=lyrs[0] ; sur_=np.asarray(surpass)
# doing some masking
    # rain_m = rain[ np.unique( sur_[0, l_idx] ), : ].copy()
    rain_m = NP_MASK.astype('f8').copy()
    rain_m[ tuple(sur_[1:3,l_idx]) ] = 0
# void points/pxls to sample from
    return np.asarray( np.where(rain_m==1) )


#~ CHOP OUT RAINFALL ABOVE MAXIMA (PER TIME.SLICE) & DISPERSE IT IN THE LAYER ~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def CHOP_MAX_RAIN( rain, NP_MASK, surpass ):#surpass = np.where(rain > MAXD_RAIN)
# https://stackoverflow.com/a/8505754/5885810
    rng = npr.default_rng()
    # the work is done in mm (no need to convert into mm/h)
    abvrain = rain[ surpass ] - MAXD_RAIN
    # total xtra rainfal per T_RES.slice
    lyrs = list(map(lambda x:np.where(surpass[0]==x)[0], np.unique(surpass[0])))
    xtra_r = np.asarray( list(map(lambda x:abvrain[ x ].sum(), lyrs)) )
    # chunks of rainfall per T_RES.slice
    xtra_n = list(map(lambda x:np.hstack([np.repeat(MAXD_RAIN *DISPERSE_, int(x)),
        np.array((x %1) *MAXD_RAIN *DISPERSE_)]), xtra_r /(MAXD_RAIN *DISPERSE_)))
# retrieving the NON-exceeding pixels (per layer) where to put surplus rain
    v_idx = list(map(lambda x:AUX_MSK(rain, np.asarray(surpass), NP_MASK, x ), lyrs))
    # computing tuples
    t_ples = list(map(lambda a,b,c:np.vstack([np.repeat(a, b),
        c[ :, rng.choice(c.shape[1], size=b, replace=False)]]),
            np.unique(surpass[0]), list(map(len, xtra_n)), v_idx))
    t_ples = tuple( np.hstack( t_ples ) )
    # convert to tuple and ADD dispersed rain
    rain[ t_ples ] = rain[ t_ples ] + np.hstack( xtra_n )
    # chop surpassed rainfall into MAXD_RAIN
    rain[ surpass ] = MAXD_RAIN
    return rain
# #~ CHOP OUT RAINFALL ABOVE MAXIMA (PER TIME.SLICE) & DISPERSE IT IN THE LAYER ~#
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# def CHOP_MAX_RAIN( rain, NP_MASK,surpass ):#surpass = np.where(rain > MAXD_RAIN)
#     while len( surpass[0] ) > 0:
#         # the work is done in mm (no need to convert into mm/h)
#         abvrain = rain[ surpass ] - MAXD_RAIN
#         # total xtra rainfal per T_RES.slice
#         xtra_r = np.asarray( list(map(lambda x:abvrain[ np.in1d(surpass[0], x) ].sum(), np.unique(surpass[0]))) )
#         # chunks of rainfall per T_RES.slice
#         xtra_n = list(map(lambda x:np.hstack([np.repeat(MAXD_RAIN *DISPERSE_, int(x)),
#             np.array((x %1) *MAXD_RAIN *DISPERSE_)]), xtra_r /(MAXD_RAIN *DISPERSE_)))
#     # doing some masking
#         rain_m = rain[ np.unique(surpass[0]), : ].sum(axis=0)
#         rain_m[ rain_m>0 ] = -1
#         rain_m[ NP_MASK==0 ] = -1
#         rain_m = rain_m +1
#     # void points/pxls to sample from
#         v_idx = np.asarray( np.where(rain_m==1) )
#         # computing tuples
#         t_ples = list(map(lambda a,b:np.vstack([np.repeat(a, b),
#             v_idx[ :, np.random.randint(0, v_idx.shape[1], b)]]) , np.unique(surpass[0]), list(map(len, xtra_n))))
#         # convert to tuple and assign dispersed rain
#         rain[ tuple( np.hstack( t_ples ) ) ] = np.hstack( xtra_n )
#         # chop surpassed rainfall into MAXD_RAIN
#         rain[ surpass ] = MAXD_RAIN
#     return rain


#~ SHUFFLES LAYERS OF PREVIOUS & CURRENT FIELDS (reads/writes to/fro NC ~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def SHUFFLING_( nreg, rain, sate, zuma, MRAIN, NP_MASK, masksum, MOTHER ):
# initialisation
    prevdate = timexx[:].data

#-!!ATTEMPT TO REMOVE ANY INFINITE/VOID LAYERS (STORED ONLY IN THE 1st.MASK/ITERATION)!!
    # if nreg > 0 and prevdate[-1] == TIMEFIL:
    if len(prevdate) != 0 and prevdate[-1] == TIMEFIL -1:
    # find the extension on the voids
        void_s = np.where(np.diff(prevdate)!=1).__getitem__(0)[-1]
    # shrink numpys
        MOTHER = MOTHER[:void_s +1]
        prevdate = prevdate[:void_s +1]
        masksum = masksum[:void_s +1]
    #** maybe there's not need to expand "masksum==tmp_suma" at this function's end?

    thisdate = np.union1d(prevdate, sate)
    tmp_suma = np.zeros(thisdate.shape)

# finding intersected dates
    # in_date = np.intersect1d(sate, prevdate, assume_unique=True)
    # intprev = np.arange(len(prevdate))[ np.in1d(prevdate, in_date, assume_unique=True) ]
    # intsate = np.arange(len(sate))[ np.in1d(sate, in_date, assume_unique=True) ]
# https://stackoverflow.com/a/74186425/5885810
    in_date, intsate, intprev = np.intersect1d(sate, prevdate, assume_unique=True, return_indices=True)

# read PREVIOUS stored rainfall into 'f8'
    # # i think python3.9.16's NETCDF4.version DOES.NOT allow the selection of VOID.indexes
    prevrain = ((ncvarx[ intprev, :].data *SCL) +ADD).round(3).astype( 'f8' ) if len(intprev)!=0 else VOIDXR.astype('f8')
# now... you can aggregate (the intersection TIME.STEPS) with current rain...
# UPDATE THIS/CURRENT RAIN (the aggregated fields are stored in CURRENT.RAIN)... & their sums
    rain[ intsate, :] = rain[ intsate, :] + prevrain
    zuma[ intsate ] = zuma[ intsate ] + masksum[ intprev ]

# CHOP OUT MAXIMA.T_RES RAINFALL in the intersections
    surpass = np.where(rain > MAXD_RAIN)
    # if len( surpass[0] ) > 0:
    while len( surpass[0] ) > 0:
        rain = CHOP_MAX_RAIN( rain, NP_MASK, surpass )
        surpass = np.where(rain > MAXD_RAIN)

# update the CUMSUM in thisdate
    this_in_prev = np.arange(len(thisdate))[ np.in1d(thisdate, prevdate, assume_unique=True) ]
    this_in_sate = np.arange(len(thisdate))[ np.in1d(thisdate, sate, assume_unique=True) ]
    tmp_suma[ this_in_prev ] = masksum
    tmp_suma[ this_in_sate ] = zuma
    cuma_tmp = np.cumsum( tmp_suma )

# HAVE I REACHED PTOT? -> (PTOT is reached in the after-step, not the before-one)
    whatsin = np.where( cuma_tmp <= MRAIN ).__getitem__(0)
    if len(cuma_tmp)-1 > whatsin[-1]:
        whatsin = np.append(whatsin, whatsin[-1] +1)

# RE-SIZING sums to MATCH NEW_DATE & MOTHER!!
    new_temp = thisdate[ whatsin ].astype( TIMEINT )
# new.UPDATED.date
#-**IT'S OUTSTANDINGLY CRUCIAL TO MERGE new_temp AS IT LOCKS PREVIOUS REACHED.FIELDS
    new_date = np.union1d(MOTHER, new_temp).astype( TIMEINT )
    sum_temp = np.zeros(new_date.shape)
    uno_ = np.arange(len(new_date))[ np.in1d(new_date, new_temp, assume_unique=True) ]
    sum_temp[ uno_ ] = tmp_suma[ whatsin ]
# new.UPDATED.suma
    tmp_suma_dos = sum_temp

# update the time.dimension -> TIMEXX[:] DOES NOT!! SHORTEN!!
    timexx[:] = new_date
    solo_uno = timexx[:].data
    """
SOLO_UNO is never gonna be SMALLER THAN -> NEW_DATE; NEW_DATE accounts for the immutable MOTHER
    """
    # if len(solo_uno) > len(new_date):
    # # CAREFUL:: you might have repeated DATES when NEW_DATE < TIMEXX[:] !!
    #     print('\n### ATTENTION!! SHRINKING HAPPENING ###')
    #     print(solo_uno)
    #     print('### ------------------------------- ###')

# zeroing the XTRA-TIMES created
#-1 [just in case]
    out_soft = len(solo_uno) - len(prevdate)
    if out_soft > 0:
        out_sid = np.arange(len(solo_uno))[ -out_soft: ]
        # ceros = ncvarx[ out_sid, : ]
        # ceros[ :, ~NP_MASK.astype('bool')==False ] = MINIMUM
        # ncvarx[ out_sid, : ] = ceros
        ZERO_FIELD( out_sid, ~NP_MASK.astype('bool') )
        # tmp_suma_dos[ out_sid ] = 0
#-2
    out_hard = len(new_date) - len(solo_uno)


    # for rr in range(len(REGIONS['npma'])):
    #     print('\n$$ STEP 0 !!:')
    #     print(f'REG_n:{nreg} -> npmask:{rr} -> MRAIN:{MRAIN}')
    #     sumas = ((ncvarx[:].data *SCL) +ADD).round(3).astype( 'f8' ).sum(axis=0)
    #     print(f'mean -> {np.ma.masked_array(sumas, mask=~REGIONS["npma"][rr].astype("bool")).mean()}\n')


# shuffling PREVDATE/array into NEW_DATE order
    # if len(solo_uno) > len(new_date):
    #     new_in_prev = np.arange(len(new_date))[ np.in1d(new_date, prevdate, assume_unique=True) ]
    #     prev_in_new = np.arange(len(prevdate))[ np.in1d(prevdate, new_date, assume_unique=True) ]
    # else:
    #     new_in_prev = np.arange(len(solo_uno))[ np.in1d(solo_uno, prevdate, assume_unique=True) ]
    #     prev_in_new = np.arange(len(prevdate))[ np.in1d(prevdate, solo_uno, assume_unique=True) ]
    new_in_prev = np.arange(len(new_date))[ np.in1d(new_date, prevdate, assume_unique=True) ]
    prev_in_new = np.arange(len(prevdate))[ np.in1d(prevdate, new_date, assume_unique=True) ]
# # (to see) what's going on?
#     print(f'new_in_prev: {new_in_prev.shape}  ||  prev_in_new: {prev_in_new.shape}\n')
#     print(f'new_date: {new_date.shape}  ||  prevdate: {prevdate.shape}\n')
# TAKE THE PREVIOUS CHUNK OF DATA & SPREAD IT ACCORDINGLY (along TIME.dim) -> "EXPAND.THE.ACCORDION"
    # ncvarx[:] DOES NOT!! SHORTEN either!!
    if len(prev_in_new) != 0:
        ncvarx[ new_in_prev, :] = ncvarx[ prev_in_new, :]

# # RE-PAINTING THE VOIDS!
#     new_notin_prev = np.arange(len(solo_uno))[ np.in1d(solo_uno, prevdate, assume_unique=True, invert=True) ]
#     # ncvarx[new_notin_prev, :] = MINIMUM


# # CODE-CHUNK TO FIND OUT WHAT'S GOING ON WHEN WRITING TO ncvarx
# # IF TESTED HERE... it's expected for the SUMS to increase as we're duplicating layers
#     for rr in range(len(REGIONS['npma'])):
#         print('\n$$ STEP 1 !!:')
#         print(f'REG_n:{nreg} -> npmask:{rr} -> MRAIN:{MRAIN}')
#         sumas = ((ncvarx[:].data *SCL) +ADD).round(3).astype( 'f8' ).sum(axis=0)
#         print(f'mean -> {np.ma.array(sumas, mask=~REGIONS["npma"][rr].astype("bool")).mean()}\n')
#         # np.ma.array(sumas, mask=~REGIONS["npma"][0].astype("bool")).mean()
#         # plt.imshow(sumas, cmap='WhiteBlueGreenYellowRed')


# shuffling SATE/array into NEW_DATE order
    # if len(solo_uno) > len(new_date):
    #     new_in_sate = np.arange(len(new_date))[ np.in1d(new_date, sate, assume_unique=True) ]
    #     # INDEXES to take from RAIN/sate
    #     sate_in_new = np.arange(len(sate))[ np.in1d(sate, new_date, assume_unique=True) ]
    # else:
    #     new_in_sate = np.arange(len(solo_uno))[ np.in1d(solo_uno, sate, assume_unique=True) ]
    #     # INDEXES to take from RAIN/sate
    #     sate_in_new = np.arange(len(sate))[ np.in1d(sate, solo_uno, assume_unique=True) ]
    new_in_sate = np.arange(len(new_date))[ np.in1d(new_date, sate, assume_unique=True) ]
    # INDEXES to take from RAIN/sate
    sate_in_new = np.arange(len(sate))[ np.in1d(sate, new_date, assume_unique=True) ]
# FIT THE CURRENT DATA INTO THE CREATED 'VOIDS' (some are intersects)
    ncvarx[ new_in_sate, :] = (( (rain[ sate_in_new, :] - ADD) / SCL ).round(0)).astype( f'{RAINFMT}' )


    # puta = rain[ sate_in_new, :].sum(axis=0)
    # np.ma.masked_array(puta, mask=~REGIONS["npma"][nreg].astype("bool")).mean()


    # for rr in range(len(REGIONS['npma'])):
    #     print('\n$$ STEP 1 !!:')
    #     print(f'REG_n:{nreg} -> npmask:{rr} -> MRAIN:{MRAIN}')
    #     sumas = ((ncvarx[:].data *SCL) +ADD).round(3).astype( 'f8' ).sum(axis=0)
    #     print(f'mean -> {np.ma.masked_array(sumas, mask=~REGIONS["npma"][rr].astype("bool")).mean()}\n')
    #     # np.ma.array(sumas, mask=~REGIONS["npma"][0].astype("bool")).mean()
    #     # plt.imshow(sumas, cmap='WhiteBlueGreenYellowRed')


# spaces to be ZEROED -> (product from bloating CUMSUMs to match NEW_DATE)
# TO BE ON THE XTRA.SAFE.SIDE?? -> [ONLY ON THE PRESENT MASK!!]
    # if len(solo_uno) > len(new_date):
    #     null_sum = np.arange(len(new_date))[ np.in1d(new_date, sate, assume_unique=True, invert=True) ]
    # else:
    #     null_sum = np.arange(len(solo_uno))[ np.in1d(solo_uno, sate, assume_unique=True, invert=True) ]
    null_sum = np.arange(len(new_date))[ np.in1d(new_date, sate, assume_unique=True, invert=True) ]
    null_sum = np.where(tmp_suma_dos==0).__getitem__(0)

    if len(null_sum) > 0:# -> because Python 3.9 (or Numpy)
        # ceros = ncvarx[ null_sum, : ]
        # ceros = ceros * ~NP_MASK.astype('bool')
        # ceros[ ceros==0 ] = MINIMUM
        # ncvarx[ null_sum, :] = ceros
        ZERO_FIELD( null_sum, ~NP_MASK.astype('bool') )


        # for rr in range(len(REGIONS['npma'])):
        #     print('\n$$ STEP 2 !!:')
        #     print(f'REG_n:{nreg} -> npmask:{rr} -> MRAIN:{MRAIN}')
        #     sumas = ((ncvarx[:].data *SCL) +ADD).round(3).astype( 'f8' ).sum(axis=0)
        #     print(f'mean -> {np.ma.array(sumas, mask=~REGIONS["npma"][rr].astype("bool")).mean()}\n')


# nullify rainfall fields beyond TOTAL.CUMSUM
    if out_hard < 0:
        # print('### TRIMMING [OUT] nc4.file ###')
    #-1:futuristic dates (to avoid repeated.time indexing)
        # MAX.Filling is equal to TIMEFIL-1 -> TIMEFIL+1==TIMEFIL
        # creating u8-arrays with np.linspace/np.arange is not so straightforward
# last_u = np.array( list(map(lambda x:np.array(x, dtype=TIMEINT), [TIMEFIL-1-x for x in range(abs(-out_hard))][::-1])) )
        last_u = np.array( [TIMEFIL -1 -x for x in range(abs(out_hard))][::-1], dtype=TIMEINT )
        tmp_xx = timexx[:]
        tmp_xx[ out_hard: ] = last_u
        timexx[:] = tmp_xx#.data
    #-2:zerox xtra.rain.fields (only in the current mask!)
        ZERO_FIELD( np.arange(len(solo_uno))[ out_hard: ], ~NP_MASK.astype('bool') )
    #-3:pad with zeros TMP_SUMA so it matches SOLO_UNO [this is proving to be useless?]
        tmp_suma_dos = np.pad(tmp_suma_dos, (0, abs(out_hard)), mode='constant')


        # for rr in range(len(REGIONS['npma'])):
        #     print('\n$$ STEP LAST !!:')
        #     print(f'REG_n:{nreg} -> npmask:{rr} -> MRAIN:{MRAIN}')
        #     sumas = ((ncvarx[:].data *SCL) +ADD).round(3).astype( 'f8' ).sum(axis=0)
        #     print(f'mean -> {np.ma.array(sumas, mask=~REGIONS["npma"][rr].astype("bool")).mean()}\n')


    return solo_uno, tmp_suma_dos, tmp_suma_dos.cumsum()[-1]


    # #-CHECK WHAT IS FASTER??--------------------------------------------------
    # r_ain = rain.copy()
    # # storing the variable as INT AND then assigning the SCL & ADD FACTORS
    # ncvarx[:] = (( (r_ain - ADD) / SCL ).round(0)).astype( f'{RAINFMT}' )
    # # or... assigning the variable as FLOAT, having previously established (in NC_FILE_II) the SCL & ADD FACTORS
    # ncvarx[:] = ROUNDX( r_ain )

    # ##-ANSWER!!---------------------------------------------------------------
    # # #%%timeit
    # # (( (r_ain - ADD) / SCL ).round(0)).astype( f'{RAINFMT}' )
    # # # 999 ms ± 50.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # # #%%timeit
    # # ROUNDX( r_ain )
    # # # 1.22 s ± 106 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

#%%

#~ "THE" LOOP CALLING THE FUNCTIONS (until CUMSUM.is.REACHED) ~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def main_loop(train, mask_shp, NP_MASK, simy, nreg, mom, M_LEN, date_origen):
# NP_MASK = REGIONS['npma'][nreg];
# train = reg_tot.copy(); mom = alltime.copy(); mask_shp = region_s['mask'].iloc[nreg]

    masksum = np.zeros_like(mom)
    # for the HAD we assume (initially) there's ~6 storm/day; then
    # ... we continue 'half-ing' the above seed
    # 30*?? ('_SF' runs); 30*?? ('_SC' runs); 30*6?? ('STANDARD' runs)
    NUM_S = 30 * 6 * M_LEN
    CUM_S = 0
    # KOUNT = 0  # TEMPORAL.STOPPING.CRITERION

    # until total rainfall is reached or no more storms to compute!
    while CUM_S < train and NUM_S >= 2:
    # while KOUNT < 3 and NUM_S >= 2:  # does the cycle 3x maximum!
#%%
        # sample random storm centres
        CENTS = scentres(mask_shp, NUM_S)  # CENTS.plot()
        # # 561 ms ± 10.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # SENTS = scentros(mask_shp, NUM_S)
        # # 1.07 s ± 29.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # CENTS.plot(region=SENTS.shp_series['geometry'], cents=SENTS.samples)

        # CENTS = scentres(mask_shp, 100)


        # find MAX_DUR
        dur_lim = np.array([T_RES, MAX_DUR]) * TIME_DICT_[TIME_OUTNC]
        DUR_S = truncated_sampling(AVGDUR[nreg], limits=dur_lim, n=NUM_S)

        # okdur == NUM_S


# calling the DATE.TIME block
        MATES, i_scaling = QUANTIZE_TIME( NUM_S, seas, simy, DUR_S[ okdur ] )




    # sampling maxima radii
        RADII = TRUNCATED_SAMPLING( RADIUS[ seas ][''], [1* MINRADIUS, None], NUM_S )
        RADII = RADII *9.5 # -> so we can have large storms than the resolution
    # polygon(s) for maximum radii
        RINGO = LAST_RING( RADII, CENTS )

    # define pandas to split the Z_bands (or not)
        qants, ztats = ZTRATIFICATION( pd.concat( RINGO ) )
    # compute copulas given the Z_bands (or not)
        MAX_I, DUR_S = list(map(np.concatenate, zip(* qants.apply( lambda x:\
            # COPULA_SAMPLING(COPULA, seas, x['E'], x['median']), axis='columns') ) ))
            COPULA_SAMPLING(COPULA, seas, x['E'], x[ Z_STAT ]), axis='columns') ) ))


    # increase/decrease maximum intensites
        # MAX_I = MAX_I[ okdur ] * (1 + STORMINESS_SC[ seas ] + ((simy +0) *STORMINESS_SF[ seas ]))
        MAX_I = MAX_I[ okdur ] * 2.1#1
    # choping any max_intensity above the permitted/designed MAXD_RAIN
        MAX_I[ MAX_I > MAXD_RAIN ] = MAXD_RAIN



    # # if FORCE_BRUTE was used -> truncation deemed necessary to avoid
    # # ...ULTRA-high intensities [chopping almost the PDF's 1st 3rd]
        # BETAS = TRUNCATED_SAMPLING( BETPAR[ seas ][''], [-0.008, +0.078], NUM_S )
        # [BETPAR[ seas ][''].cdf(x) if x else None for x in [-.035, .035]]
        BETAS = RANDOM_SAMPLING( BETPAR[ seas ][''], NUM_S )

    # multiply and displace storm.centres
        M_CENTS = MOVING_STORM( CENTS[ okdur ], i_scaling, seas )

    # expand supporting arrays
        M_RADII, M_MAXI, M_DURS, M_BETAS = XPAND(
            [RADII[ okdur ], MAX_I, DUR_S[ okdur ], BETAS], i_scaling )

    # compute granular rainfall over intermediate rings
        # RINGS = LOTR( RADII[ okdur ], MAX_I, DUR_S[ okdur ], BETAS, CENTS[ okdur ] )
        RINGS = LOTR( M_RADII, M_MAXI, M_DURS, M_BETAS, M_CENTS )

    # COLLECTING THE STORMS
        # STORM_MATRIX = list(map(RASTERIZE, RINGS, RINGO[ okdur ]))
        STORM_MATRIX = list(map(RASTERIZE, RINGS ))

    # returns a time-sorted & void.trimmed rainfall cube
        rain, sate, zuma = RAIN_CUBO( STORM_MATRIX, np.concatenate(i_scaling), MATES, NP_MASK )
#%%
        # NEWMOM=solo_uno; summask=tmp_suma_dos; CUM_S=tmp_suma_dos.cumsum()[-1]
        NEWMOM, summask, CUM_S = SHUFFLING_( nreg, rain, sate, zuma, MRAIN, NP_MASK, masksum, MOTHER )
        # NEWMOM, masksum, CUM_S, acttim = SHUFFLING_( nreg, rain, sate, zuma, MRAIN, NP_MASK, masksum, MOTHER )

# # stupid output checking
#         print(f'\n CUMsum - PTOT: {CUM_S -MRAIN}\n----')

        masksum = summask

    # 'decreasing the counter'
        # NUM_S = int(NUM_S /2)#/1.5)
        NUM_S = int(NUM_S *.9)
        # KOUNT += 1

# WARN IF THERE IS NO CONVERGING
    assert not (CUM_S < MRAIN and NUM_S < 2), f'Iteration for '\
        f'{nsim+1}: YEAR {simy} not converging!!\nTry a larger initial '\
        'seed (i.e., variable "NUM_S"). If the problem persists, it '\
        'might be very likely that the catchment (stochastic) '\
        'parameterization is not adequate.' # the more you increase the slower it gets!!

    # print(f'\n pxls in mask: {NP_MASK.sum()}\n----')
    return NEWMOM, CUM_S


# %% central wrapper

def STARM():
    PDFS = read_pdfs()
    construct_pdfs(PDFS)
    year_zero, M_LEN, DATE_POOL = wet_days()
    date_origen = datetime.strptime(DATE_ORIGIN, '%Y-%m-%d').replace(
        tzinfo=ZoneInfo(TIME_ZONE))
    SPACE = masking()
    region_s = regionalisation(SPACE)
    # CREATE NC.FILE
    nc = nc4.Dataset('./model_output/RUN_TOAST_X.nc', 'w', format='NETCDF4')
    nc.created_on = datetime.now(tzlocal()).strftime('%Y-%m-%d %H:%M:%S %Z')#%z

    print('\nRUN PROGRESS')
    print('************')

    # FOR EVERY SIMULATION
    for nsim in range(NUMSIMS):  # nsim = 0
        print(f'\tRUN: {"{:02d}".format(nsim + 1)}/{"{:02d}".format(NUMSIMS)}')
        # 1ST FILL OF THE NC.FILE (defining global vars & CRS)
        sub_grp, tag_y, tag_x = nc_file_i(nc, nsim, SPACE)
        # FOR EVERY YEAR of the SIMULATION
        for simy in tqdm(range(NUMSIMYRS), ncols=50):  # simy = 0
            # 2ND FILL OF THE NC.FILE (creating the TIME & RAIN vars)
            # timexx, ncvarx = nc_file_ii(
            sub_grp = nc_file_ii(
                sub_grp, simy, year_zero, date_origen, tag_y, tag_x,
                )
    nc.close()


def STORM( NC_NAMES ):
    global cut_lab, cut_bin, nc, sub_grp, nsim

# define Z_CUTS labelling (if necessary)
    if Z_CUTS:
        cut_lab = [f'Z{x+1}' for x in range(len(Z_CUTS) +1)]
        cut_bin = np.union1d(Z_CUTS, [0, 9999])

    # read (and check) the PDF-parameters
    PDFS = read_pdfs()
    # PDFS = read_pdfs('./model_input/ProbabilityDensityFunctions_OND_1r.csv')
    construct_pdfs(PDFS)

    # define some xtra-basics
    year_zero, M_LEN, DATE_POOL = wet_days()
    # convert DATE_ORIGIN into 'datetime'
    # https://stackoverflow.com/a/623312/5885810
    # https://stackoverflow.com/q/70460247/5885810  (timezone no pytz)
    # https://stackoverflow.com/a/65319240/5885810  (replace timezone)
    date_origen = datetime.strptime(DATE_ORIGIN, '%Y-%m-%d').replace(
        tzinfo=ZoneInfo(TIME_ZONE))

    # imported CLASS (from PDFS_) definining GRID.settings
    # BUFFRX_MASK --> SPACE.buffer_mask
    # CATCHMENT_MASK --> SPACE.catchment_mask
    # BUFFRX --> "might not be necessary; otherwise make it visible in PDFS_.masking"
    # XS -> SPACE.xs  |  YS -> SPACE.ys
    # llim -> SPACE.bbbox['l']  |  rlim -> SPACE.bbbox['r']
    # blim -> SPACE.bbbox['b']  |  tlim -> SPACE.bbbox['t']
    SPACE = masking()  # SPACE.plot()
    # SPACE = masking(catchment=SHP_FILE)

    # # rain_fs = READ_REALIZATION( RAIN_MAP, SUBGROUP, WKT_OGC, YS, XS )
    # # REGIONS = REGIONALISATION( rain_fs, NREGIONS, BUFFRX_MASK, CATCHMENT_MASK )
    # rain_fs = READ_REALIZATION(RAIN_MAP, '', WKT_OGC, SPACE.ys, SPACE.xs)
    # REGIONS = REGIONALISATION(rain_fs, NREGIONS, SPACE.buffer_mask, SPACE.catchment_mask)

    # region_s = regionalisation(ZON_FILE.replace('.shp', f'_{SEASON_TAG}_{9}r.shp'), 'region', SPACE,)
    region_s = regionalisation(
        ZON_FILE.replace('.shp', f'_{SEASON_TAG}_{NREGIONS}r.shp'),
        'region', SPACE,
        )
    # plt.imshow(region_s['kmeans'], interpolation='none', cmap='turbo')
    # plt.imshow(region_s['npma'][-1], interpolation='none', cmap='plasma_r')

    # sampling/updating total seasonal rainfall (a.k.a., STOPPING CRITERIA!)
    # seasonal_rain_e(TOTALP[-1], n=10)
    if ptot_or_kmean == 1:
        region_s['rain'] = pd.Series(np.ravel(list(map(
            lambda x: truncated_sampling(TOTALP[x], limits=(NO_RAIN, np.inf)),
            range(len(TOTALP))))), name='s_rain',)

# define transformation constants/parameters for INTEGER rainfall-NC-output
    SCL, ADD, MINIMUM = nc_bytes()

    VOIDXR = np.empty((0, len(SPACE.ys), len(SPACE.xs))).astype(f'{RAINFMT}')
    VOIDXR.fill(np.round(0, 0).astype(f'{RAINFMT}'))
    # VOIDXR.fill(np.round((0 - ADD) / SCL, 0).astype(f'{RAINFMT}'))


    # # https://gis.stackexchange.com/a/267326/127894  (get EPSG/CRS from raster)
    # from osgeo import osr
    # tIFF = gdal.Open(abspath(join(parent_d, DEM_FILE)))
    # tIFF_proj = osr.SpatialReference( wkt=tIFF.GetProjection() ).GetAttrValue('AUTHORITY', 1)
    # tIFF = None

    # CREATE NC.FILE
    nc = nc4.Dataset(NC_NAMES, 'w', format='NETCDF4')
    nc.created_on = datetime.now(tzlocal()).strftime('%Y-%m-%d %H:%M:%S %Z')#%z

    print('\nRUN PROGRESS')
    print('************')

    # FOR EVERY SIMULATION
    for nsim in range(NUMSIMS):  # nsim=0
        print(f'\tRUN: {"{:02d}".format(nsim + 1)}/{"{:02d}".format(NUMSIMS)}')

        # 1ST FILL OF THE NC.FILE (defining global vars & CRS)
        sub_grp, tag_y, tag_x = nc_file_i(nc, nsim, SPACE)

        # FOR EVERY YEAR of the SIMULATION
        for simy in tqdm(range(NUMSIMYRS), ncols=50):  # simy=0
            # 2ND FILL OF THE NC.FILE (creating the TIME & RAIN vars)
            # timexx, ncvarx = nc_file_ii(
            sub_grp = nc_file_ii(
                sub_grp, simy, year_zero, date_origen, tag_y, tag_x,
                )

            # a void array/variable cannot traverse/GLOBAL
            alltime = np.array([], dtype='f8')
            cum_out = []

            # FOR EVERY N_REGION
            for nreg in tqdm(range(NREGIONS), ncols=50):  # nreg=0

                # print(f'\nNREGIONS: {nreg + 1}/{NREGIONS}')

                # scale (or not) the total seasonal rainfall
                reg_tot = region_s['rain'].iloc[nreg] *\
                    (1 + PTOT_SC) * (1 + ((simy + 1) * PTOT_SF))

                newmom, cumout = main_loop(reg_tot,
                    region_s['mask'].iloc[nreg], region_s['npma'][nreg],
                    simy, nreg, alltime, M_LEN, date_origen,
                    )
                alltime = newmom
                cum_out.append( cumout )




# # NO NEED for this.updating NO MORE!
#             # add/pass SEAS and SIMY to the REGIONS.dictionary
#                 # https://stackoverflow.com/a/209854/5885810
#                 # https://www.freecodecamp.org/news/add-to-dict-in-python/
#                 REGIONS.update( dict(zip( ['seas','simy'],
#                     list(map(lambda x:[x] *len(REGIONS['rain']), [seas, simy])) )) )



        # storing MEANs as INTs
            cum_xtra = []
            for rr in range(len(REGIONS['npma'])):
                sumas = ((ncvarx[:].data *SCL) +ADD).round(3).astype('f8').sum(axis=0)
                # sumas = ncvarx[:].data.astype('f8').sum(axis=0)
                cum_xtra.append( np.ma.array(sumas, mask=~REGIONS["npma"][rr].astype("bool")).mean() )

            if RAINFMT[0]!='f':
            # you'd have PROBLEMS.IF doing this assignation BEFORE filling entirely the NCVARX.var
                ncvarx.scale_factor = SCL
                ncvarx.add_offset = ADD

        # just to fill "maskxx" only once (we don't need re-regionalisation every run, do we?)
            maskxx[:] = REGIONS['kmeans']
        # temporal xport of kmeans+regions
            pd.DataFrame({'k':range(NREGIONS), 'mean_in':REGIONS['rain'], 'mean_out':cum_OUT
                          ,'mean_xtra':cum_xtra
                }).to_csv(NC_NAMES.replace('.nc','_kmeans.csv'), index=False, mode='a', sep=',')

    nc.close()

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#-  CORE COMPUTATION & NC.FILE FILLING --------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#%%

if __name__ == '__main__':

    # from check_input import WELCOME
    # NC_NAMES = WELCOME()

    # NC_NAMES = './model_output/RUN_240909T1017_ptotT-_stormsS+_TOAST.nc'

    # STORM(NC_NAMES)
    STARM()
