#%% CALLING LIBRARIES
"""
this RAINFALL.py belongs to STORM3
"""

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
# https://github.com/slundberg/shap/issues/2909    (suppresing the one from numba 0.59.0)
warnings.filterwarnings('ignore', message=".*The 'nopython' keyword.*")

# https://stackoverflow.com/a/248066/5885810
from os.path import abspath, dirname, join
parent_d = dirname(__file__)    # otherwise, will append the path.of.the.tests
# parent_d = './'               # to be used in IPython

import numpy as np
import pandas as pd
# # https://stackoverflow.com/a/65562060/5885810  (ecCodes in WOS)
import xarray as xr
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

from pointpats import random as pran
# import libpysal as ps
# from pointpats import PoissonPointProcess, Window#, PointPattern

from chunking import CHUNK_3D
from parameters import *
from realization import READ_REALIZATION, REGIONALISATION, EMPTY_MAP

# """ only necessary if Z_CUTS & SIMULATION used """
# from rasterstats import zonal_stats
""" STORM3.0 ALSO runs WITHOUT this library!!! """
import vonMisesMixtures as vonmises


#~ INSTALLING THE vonMisesMixtures PACKAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
# https://framagit.org/fraschelle/mixture-of-von-mises-distributions
first try this(in the miniconda/anacoda prompt):

    pip install vonMisesMixtures

...but as of July 21, 2022, it didn't work (not available in PIP no more?)
otherwise, proceed as the above link suggests:

    git clone https://framagit.org/fraschelle/mixture-of-von-mises-distributions.git
    cd mixture-of-von-mises-distributions/
    pip install .

do the 'cloning' in your library environment, i.e.,
path-to-miniconda3//envs/prll/lib/python3.10/site-packages  (linux)
path-to-miniconda3\\envs\py39\Lib\site-packages             (windows)
that's it you're all set now!
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
STORM [STOchastic Rainfall Model] produces realistic regional or watershed rainfall under various
climate scenarios based on empirical-stochastic selection of historical rainfall characteristics.

Based on Rios...
[ https://doi.org/10.1088/1748-9326/aa8e50 ]

version name: STORM3

Authors:
    Manuel F. Rios Gaona 2023
Date created : 2023/05/11
Last modified: 2023/09/06
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

# PRE_FILE = './model_input/ProbabilityDensityFunctions_ONE--ANALOG.csv'      # output from 'pre_processing.py'
# PRE_FILE = './model_input/ProbabilityDensityFunctions_ONE--ANALOG-pmf.csv'  # output from 'pre_processing.py'
SHP_FILE = './model_input/HAD_basin.shp'                # catchment shape-file in WGS84
# DEM_FILE = './model_input/dem/WGdem_wgs84.tif'        # aoi raster-file (optional**)
# DEM_FILE = './model_input/dem/WGdem_26912.tif'        # aoi raster-file in local CRS (***)
DEM_FILE = None
OUT_PATH = './model_output'                             # output folder

# RAIN_MAP = '../CHIMES/3B-HHR.MS.MRG.3IMERG.20101010-S100000-E102959.0600.V06B.HDF5'     # no.CRS at all!
# RAIN_MAP = './realisation_MAM_crs-wrong.nc'           # no..interpretable CRS
RAIN_MAP = './realisation_MAM_crs-OK.nc'                # yes.interpretable CRS
SUBGROUP = ''
CLUSTERS = 1#4#1                                        # number of regions to split the whole.region into

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

# #~DRYP.WAY~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# # USE ANY OF THE FOLLOWING SET OF PARAMETERS, IF USING "SHP_REGION_GRID() -> [GRID.based 1ST]"
# # DRYP actual/current grid
# XLLCORNER =  1319567.308750340249                       # in meters! (x.coord of the lower.left edge, i.e., not.the.pxl.center)
# YLLCORNER = -1170429.328196450602                       # in meters! (y.coord of the lower.left edge, i.e., not.the.pxl.center)
# X_RES     =      919.241896152628                       # in meters! (pxl.resolution for the 'regular/local' CRS)
# Y_RES     =      919.241896152628                       # in meters! (pxl.resolution for the 'regular/local' CRS)
# N_X       =     2313                                    # number of cells/pxls in the X-axis
# N_Y       =     2614                                    # number of cells/pxls in the Y-axis
# PARAMETERS for a HAD-tight-adjusted GRID ( 5km res)
BUFFER    =     8000.                                   # in meters! -> buffer distance (out of the HAD)
X_RES     =     5000.                                   # in meters! (pxl.resolution for the 'regular/local' CRS)
Y_RES     =     5000.                                   # in meters! (pxl.resolution for the 'regular/local' CRS)
# XLLCORNER =  1350000. -np.ceil(BUFFER /X_RES) *X_RES    # in meters! (x.coord of the lower.left edge, i.e., not.the.pxl.center)
# YLLCORNER = -1165000. -np.ceil(BUFFER /Y_RES) *Y_RES    # in meters! (y.coord of the lower.left edge, i.e., not.the.pxl.center)
# N_X       =  int(405  +np.ceil(BUFFER /X_RES)  *2)      # number of cells/pxls in the X-axis
# N_Y       =  int(464  +np.ceil(BUFFER /Y_RES)  *2)      # number of cells/pxls in the Y-axis
# from: https://stackoverflow.com/a/62264948/5885810
XLLCORNER =  1350000. -(-1 *(BUFFER /X_RES) //1 *-1) *X_RES    # in meters! (x.coord of the lower.left edge, i.e., not.the.pxl.center)
YLLCORNER = -1165000. -(-1 *(BUFFER /Y_RES) //1 *-1) *Y_RES    # in meters! (y.coord of the lower.left edge, i.e., not.the.pxl.center)
N_X       =  int(405  +(-1 *(BUFFER /X_RES) //1 *-1)  *2)      # number of cells/pxls in the X-axis
N_Y       =  int(464  +(-1 *(BUFFER /Y_RES) //1 *-1)  *2)      # number of cells/pxls in the Y-axis
# # PARAMETERS for a HAD-tight-adjusted GRID (10km res)
# BUFFER    =     8000.                                   # in meters! -> buffer distance (out of the HAD)
# X_RES     =    10000.                                   # in meters! (pxl.resolution for the 'regular/local' CRS)
# Y_RES     =    10000.                                   # in meters! (pxl.resolution for the 'regular/local' CRS)
# XLLCORNER =  1350000. -np.ceil(BUFFER /X_RES) *X_RES    # in meters! (x.coord of the lower.left edge, i.e., not.the.pxl.center)
# YLLCORNER = -1170000. -np.ceil(BUFFER /Y_RES) *Y_RES    # in meters! (y.coord of the lower.left edge, i.e., not.the.pxl.center)
# N_X       =  int(203  +np.ceil(BUFFER /X_RES)  *2)      # number of cells/pxls in the X-axis
# N_Y       =  int(233  +np.ceil(BUFFER /Y_RES)  *2)      # number of cells/pxls in the Y-axis
# #~DRYP.WAY~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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


#%% FUNCTIONS' DEFINITION

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- SET UP SPACE-TIME DOMAIN & UPDATE PARAMETERS ---------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ replace FILE.PARAMETERS with those read from the command line ~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def PAR_UPDATE( args ):
    for x in list(vars( args ).keys()):
# https://stackoverflow.com/a/2083375/5885810   (exec global... weird)
        exec(f'globals()["{x}"] = args.{x}')
    # print([PTOT_SC, PTOT_SF])


#~ DEFINE THE DAYS OF THE SEASON (to 'sample' from) ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def WET_SEASON_DAYS():
    global SEED_YEAR, M_LEN, DATE_POOL, DOY_POOL, DATE_ORIGEN
    SEED_YEAR = SEED_YEAR if SEED_YEAR else datetime.now().year
# which season is None/void/null
    mvoid = list(map(lambda x:None in x, zip(SEASONS_MONTHS)))
# transform months into numbers (if passed as strings)
    month = list(map( lambda m,v: None if v else\
        list(map(lambda m:m if type(m) == int else datetime.strptime(m,'%b').month, m)),
        SEASONS_MONTHS, mvoid ))
# compute monthly duration (12 months in a year)
    M_LEN = [None if v else \
        [1+m[1]-m[0] if m[1]-m[0]>=0 else 1+12+m[1]-m[0]] for m,v in zip(month, mvoid)]
# construct the date.times & update their years
    DATE_POOL = [None if v else \
        [datetime(year=SEED_YEAR,month=m[0],day=1),
          datetime(year=SEED_YEAR,month=m[0],day=1) + relativedelta(months=l[0])] \
            for m,l,v in zip(month, M_LEN, mvoid)]
    for i in range(len(DATE_POOL))[1:]:
        DATE_POOL[i] = None if mvoid[i] else \
            [DATE_POOL[i][0].replace(year=DATE_POOL[i-1][-1].year),
              DATE_POOL[i][0].replace(year=DATE_POOL[i-1][-1].year)\
                  + relativedelta(months=M_LEN[i][0])]
# extract Day(s)-Of-Year(s)
# https://stackoverflow.com/a/623312/5885810
    DOY_POOL = list(map(lambda v,d: None if v else
        list(map(lambda d: d.timetuple().tm_yday, d)), mvoid, DATE_POOL ))
# convert DATE_ORIGIN into 'datetime' (just to not let this line hanging out all alone)
# https://stackoverflow.com/q/70460247/5885810  (timezone no pytz)
# https://stackoverflow.com/a/65319240/5885810  (replace timezone)
    DATE_ORIGEN = datetime.strptime(DATE_ORIGIN, '%Y-%m-%d').replace(
        tzinfo=ZoneInfo(TIME_ZONE))


#~ WORKING OUT THE CATCHMENT (& ITS BUFFER) MASK(S) [GRID.based 1ST] ~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def SHP_REGION_GRID():
    """
    # DRYP actual/current grid
    XLLCORNER =  1319567.308750340249       # in meters! (x.coord of the lower.left edge, i.e., not.the.pxl.center)
    YLLCORNER = -1170429.328196450602       # in meters! (y.coord of the lower.left edge, i.e., not.the.pxl.center)
    X_RES     =      919.241896152628       # in meters! (pxl.resolution for the 'regular/local' CRS)
    Y_RES     =      919.241896152628       # in meters! (pxl.resolution for the 'regular/local' CRS)
    N_X       =     2313                    # number of cells/pxls in the X-axis
    N_Y       =     2614                    # number of cells/pxls in the Y-axis
    """
    global llim, rlim, blim, tlim, XS, YS, CATCHMENT_MASK, BUFFRX, BUFFRX_MASK
# infering (and rounding) the limits of the buffer-zone
    # # this way COULD ALSO be... but it's not what DRYP.ASC means
    # llim = XLLCORNER - X_RES *1/2
    # rlim = XLLCORNER + X_RES *(N_X -1/2)
    # blim = YLLCORNER - Y_RES *1/2
    # tlim = YLLCORNER + Y_RES *(N_Y -1/2)
    llim = XLLCORNER
    rlim = XLLCORNER + X_RES *N_X
    blim = YLLCORNER
    tlim = YLLCORNER + Y_RES *N_Y
# read WG-catchment shapefile (assumed to be in WGS84)
    wtrwgs = gpd.read_file( abspath( join(parent_d, SHP_FILE) ) )
# transform it into EPSG:42106 & update & the buffer
# https://gis.stackexchange.com/a/328276/127894 (geo series into gpd)
    # wtrshd = wtrwgs.to_crs( epsg=42106 )
    wtrshd = wtrwgs.to_crs( crs = WKT_OGC )          # //epsg.io/42106.wkt
# this assumes for the edges of the grid to engulf the catchment.shape
    bufdif = np.array([-(llim -wtrshd.bounds.minx[0]), (rlim -wtrshd.bounds.maxx[0]),
        -(blim -wtrshd.bounds.miny[0]), (tlim -wtrshd.bounds.maxy[0])])
# is the catchment.SHP fully within the GRID.bounds??
    assert not (bufdif <0).any(), f'Cathment.SHP out of boundaries!\nPlease, '\
        f'ensure that the GRID.edges fully engulf the catchment.SHP.'
    BUFFER = np.floor( bufdif.min() )
    BUFFRX = gpd.GeoDataFrame(geometry=wtrshd.buffer( BUFFER ))#.to_crs(epsg=4326)
#~IN.CASE.YOU.WANNA.XPORT.(OR.USE).THE.MASK+BUFFER.as.geoTIFF~~~~~~~~~~~~~~~~~~#
    # # ACTIVATE if IN.TIFF
    # tmp_file = 'tmp-raster_mask-buff_GRID.tif'
    # tmp = gdal.Rasterize(tmp_file, BUFFRX.to_json(), format='GTiff'
    # ACTIVATE if IN.MEMORY
    tmp = gdal.Rasterize('', BUFFRX.to_json(), format='MEM'#, add=0
        , xRes=X_RES, yRes=Y_RES, noData=0, burnValues=1, allTouched=True
        , outputType=gdal.GDT_Int16, outputBounds=[llim, blim, rlim, tlim]
        # , targetAlignedPixels=True
        , targetAlignedPixels=False # (check: https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap)
    # UPDATE needed for outputSRS [in WKT instead of PROJ4]
        , outputSRS=pp.CRS.from_wkt(WKT_OGC).to_proj4()
        # # , width=(abs(rlim-llim)/X_RES).astype('u2'), height=(abs(tlim-blim)/X_RES).astype('u2')
        )
    BUFFRX_MASK = tmp.ReadAsArray().astype('u1')
    tmp = None
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# #~XPORT.IT.as.NUMPY~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     np.save('tmp-raster_mask-buff_GRID', BUFFRX_MASK.astype('u1'), allow_pickle=True, fix_imports=True)
# #~DOING PARQUET (awesome compression at this RES!)~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     # after: https://stackoverflow.com/a/68760276/5885810
#     import pyarrow as pa
#     import pyarrow.parquet as pq
#     # create one arrow array per column
#     arrax = list(map(pa.array, BUFFRX_MASK.astype('u1')))
#     table = pa.Table.from_arrays(arrax, names=list(map(str, range(len(arrax))))) # give names to each columns
#     # xport it
#     pq.write_table(table, 'tmp-raster_mask-buff_GRID.pq')
# # read it back as numpy (through pandas):
#     tabpq = pq.read_table('tmp-raster_mask-buff_GRID.pq')
#     tespq = tabpq.to_pandas().T.to_numpy()
# # #~some.PLOTTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     import matplotlib.pyplot as plt
#     plt.imshow(var, interpolation='none')
#     plt.show()
#     # OR
#     from rasterio.plot import show
#     from rasterio import open as ropen
#     tmp_file = 'tmp-raster.tif'
#     srcras = ropen(tmp_file)
#     fig, ax = plt.subplots()
#     ax = show(srcras, ax=ax, cmap='viridis', extent=[
#         srcras.bounds[0], srcras.bounds[2], srcras.bounds[1], srcras.bounds[3]])
#     srcras.close()
# #~and.some.XTRA.TESTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    # # https://gis.stackexchange.com/q/344942/127894   (flipped raster)
#    # ds = gdal.Open('tmp-raster.tif')
#    # gt = ds.GetGeoTransform()
#    # if gt[2] != 0.0 or gt[4] != 0.0: print ('file is not stored with north up')
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# BURN THE CATCHMENT SHP INTO RASTER (WITH CATCHMENT-BUFFER EXTENSION)
# https://stackoverflow.com/a/47551616/5885810  (idx polygons intersect)
# https://gdal.org/programs/gdal_rasterize.html
# https://lists.osgeo.org/pipermail/gdal-dev/2009-March/019899.html (xport ASCII)
# https://gis.stackexchange.com/a/373848/127894 (outputing NODATA)
# https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap (targetAlignedPixels==True)
    tmp = gdal.Rasterize('', wtrshd.to_json(), format='MEM', add=0
        , xRes=X_RES, yRes=Y_RES, noData=0, burnValues=1, allTouched=True
        , outputType=gdal.GDT_Int16, outputBounds=[llim, blim, rlim, tlim]
        # , targetAlignedPixels=True
        , targetAlignedPixels=False # (check: https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap)
    # UPDATE needed for outputSRS [in WKT instead of PROJ4]
        , outputSRS=pp.CRS.from_wkt(WKT_OGC).to_proj4()
        # # , width=(abs(rlim-llim)/X_RES).astype('u2'), height=(abs(tlim-blim)/X_RES).astype('u2')
        )
    """
don't i need to use "targetAlignedPixels=True" here in this case??
    """
    CATCHMENT_MASK = tmp.ReadAsArray().astype('u1')
    tmp = None           # flushing!
# #~some.more.PLOTTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     import matplotlib.pyplot as plt
#     plt.imshow(CATCHMENT_MASK, interpolation='none', aspect='equal', origin='upper',
#         cmap='nipy_spectral_r', extent=(llim, rlim, blim, tlim))
#     plt.show()
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# #~IN.CASE.YOU.WANNA.XPORT.THE.MASK.as.ASCII~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     # CORRECT.way (as it registers what is NODATA)
#     tmp = gdal.Rasterize(''
#         , wtrshd.to_json(), xRes=X_RES, yRes=Y_RES, allTouched=True, initValues=-9999., burnValues=1., noData=-9999.
#         , outputType=gdal.GDT_Float32, targetAlignedPixels=False/True, outputBounds=[llim, blim, rlim, tlim]# , add=0
#         , outputSRS=pp.CRS.from_wkt(WKT_OGC).to_proj4(), format='MEM')
#     # INCORRECT.way (but is consistent with DRYP.ASC)
#     tmp = gdal.Rasterize(''
#         , wtrshd.geometry.to_json(), xRes=X_RES, yRes=Y_RES, allTouched=True, initValues=-9999., burnValues=1.
#         , outputType=gdal.GDT_Float32, targetAlignedPixels=False, outputBounds=[llim, blim, rlim, tlim]
#         # , outputSRS=pp.CRS.from_wkt(WKT_OGC).to_proj4()
#         , format='MEM')
#     CATCHMENT_MASK = tmp.ReadAsArray()
#     tmv_file = 'tmp-raster.asc'
#     tmv = gdal.GetDriverByName( 'AAIGrid' ).CreateCopy(tmv_file, tmp)
#     tmv = None           # flushing!
#     tmp = None           # flushing!
#     import os
#     os.unlink(f"./{tmv_file.replace('.asc','.prj')}")
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# #~XPORT.IT.as.NUMPY~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     np.save('tmp-raster_mask_GRID', CATCHMENT_MASK.astype('u1'), allow_pickle=True, fix_imports=True)
# #~XPORT.IT.as.PICKLE [but don't use it for NUMPYs!]~~~~~~~~~~~~~~~~~~~~~~~~~~#
# # https://stackoverflow.com/a/62883390/5885810
#     import pickle
#     with open('tmp-raster_mask_GRID.pkl','wb') as f: pickle.dump(CATCHMENT_MASK.astype('u1'), f)
# #~DOING PARQUET (awesome compression at this RES!)~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# # after: https://stackoverflow.com/a/68760276/5885810
#     import pyarrow as pa
#     import pyarrow.parquet as pq
#     # create one arrow array per column
#     arrax = list(map(pa.array, CATCHMENT_MASK.astype('u1')))
#     table = pa.Table.from_arrays(arrax, names=list(map(str, range(len(arrax))))) # give names to each columns
#     # xport it
#     pq.write_table(table, 'tmp-raster_mask_GRID.pq')
# # read it back as numpy (through pandas):
#     tabpq = pq.read_table('tmp-raster_mask_GRID.pq')
#     tespq = tabpq.to_pandas().T.to_numpy()
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# DEFINE THE COORDINATES OF THE XY.AXES
    XS, YS = list(map( lambda a,b,c: np.arange(a +c/2, b +c/2, c),
                      [llim,blim],[rlim,tlim],[X_RES,Y_RES] ))
# flip YS??
    YS = np.flipud( YS )      # -> important...so rasters are compatible with numpys


#~ WORKING OUT THE CATCHMENT (& ITS BUFFER) MASK(S) [BUFFER.based 1ST] ~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def SHP_REGION():
    """
    X_RES     =     5000.                   # in meters! (pxl.resolution for the 'regular/local' CRS)
    Y_RES     =     5000.                   # in meters! (pxl.resolution for the 'regular/local' CRS)
    """
    global llim, rlim, blim, tlim, XS, YS, CATCHMENT_MASK, BUFFRX, BUFFRX_MASK
# read WG-catchment shapefile (assumed to be in WGS84)
    wtrwgs = gpd.read_file( abspath( join(parent_d, SHP_FILE) ) )
# transform it into EPSG:42106 & make the buffer    # this code does NOT work!
# https://gis.stackexchange.com/a/328276/127894     (geo series into gpd)
    # wtrshd = wtrwgs.to_crs( epsg=42106 )
    wtrshd = wtrwgs.to_crs( crs = WKT_OGC )          # //epsg.io/42106.wkt
    BUFFRX = gpd.GeoDataFrame(geometry=wtrshd.buffer( BUFFER ))#.to_crs(epsg=4326)
# infering (and rounding) the limits of the buffer-zone
    llim = np.floor( BUFFRX.bounds.minx[0] /X_RES ) *X_RES #+X_RES/2
    rlim = np.ceil(  BUFFRX.bounds.maxx[0] /X_RES ) *X_RES #-X_RES/2
    blim = np.floor( BUFFRX.bounds.miny[0] /Y_RES ) *Y_RES #+Y_RES/2
    tlim = np.ceil(  BUFFRX.bounds.maxy[0] /Y_RES ) *Y_RES #-Y_RES/2

#~IN.CASE.YOU.WANNA.XPORT.(OR.USE).THE.MASK+BUFFER.as.geoTIFF~~~~~~~~~~~~~~~~~~#
    # # ACTIVATE if IN.TIFF
    # tmp_file = 'tmp-raster_mask-buff.tif'
    # tmp = gdal.Rasterize(tmp_file, BUFFRX.to_json(), format='GTiff'
    # ACTIVATE if IN.MEMORY
    tmp = gdal.Rasterize('', BUFFRX.to_json(), format='MEM'#, add=0
        , xRes=X_RES, yRes=Y_RES, noData=0, burnValues=1, allTouched=True
        , outputType=gdal.GDT_Int16, outputBounds=[llim, blim, rlim, tlim]
        , targetAlignedPixels=True
        # , targetAlignedPixels=False # (check: https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap)
    # UPDATE needed for outputSRS [in WKT instead of PROJ4]
        , outputSRS=pp.CRS.from_wkt(WKT_OGC).to_proj4()
        # # , width=(abs(rlim-llim)/X_RES).astype('u2'), height=(abs(tlim-blim)/X_RES).astype('u2')
        )
    BUFFRX_MASK = tmp.ReadAsArray().astype('u1')
    tmp = None
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# #~XPORT.IT.as.NUMPY~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     np.save('tmp-raster_mask-buff', BUFFRX_MASK.astype('u1'), allow_pickle=True, fix_imports=True)
# #~XPORT.IT.as.PICKLE [but don't use it for NUMPYs!]~~~~~~~~~~~~~~~~~~~~~~~~~~#
# # https://stackoverflow.com/a/62883390/5885810
#     import pickle
#     with open('tmp-raster_mask-buff.pkl','wb') as f: pickle.dump(BUFFRX_MASK.astype('u1'), f)
# # read it back as numpy (through pandas)
#     with open('tmp-raster_mask-buff.pkl', 'rb') as db_file:
#         db_pkl = pickle.load( db_file )
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# #~some.PLOTTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     import matplotlib.pyplot as plt
#     plt.imshow(BUFFRX_MASK, interpolation='none')
#     plt.show()
#     # OR
#     from rasterio.plot import show
#     from rasterio import open as ropen
#     tmp_file = 'tmp-raster.tif'
#     srcras = ropen(tmp_file)
#     fig, ax = plt.subplots()
#     ax = show(srcras, ax=ax, cmap='viridis', extent=[
#         srcras.bounds[0], srcras.bounds[2], srcras.bounds[1], srcras.bounds[3]])
#     srcras.close()
# #~and.some.XTRA.TESTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     # # https://gis.stackexchange.com/q/344942/127894   (flipped raster)
#     # ds = gdal.Open('tmp-raster.tif')
#     # gt = ds.GetGeoTransform()
#     # if gt[2] != 0.0 or gt[4] != 0.0: print ('file is not stored with north up')
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~BURN THE CATCHMENT SHP INTO RASTER (WITHOUT BUFFER EXTENSION)~~~~~~~~~~~~~~~~#
# https://stackoverflow.com/a/47551616/5885810  (idx polygons intersect)
# https://gdal.org/programs/gdal_rasterize.html
# https://lists.osgeo.org/pipermail/gdal-dev/2009-March/019899.html (xport ASCII)
# https://gis.stackexchange.com/a/373848/127894 (outputing NODATA)
# https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap (targetAlignedPixels==True)
    tmp = gdal.Rasterize('', wtrshd.to_json(), format='MEM', add=0
        , xRes=X_RES, yRes=Y_RES, noData=0, burnValues=1, allTouched=True
        , outputType=gdal.GDT_Int16, outputBounds=[llim, blim, rlim, tlim]
        , targetAlignedPixels=True
        # ,targetAlignedPixels=False # (check: https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap)
    # UPDATE needed for outputSRS [in WKT instead of PROJ4]
        , outputSRS=pp.CRS.from_wkt(WKT_OGC).to_proj4()
        # # , width=(abs(rlim-llim)/X_RES).astype('u2'), height=(abs(tlim-blim)/X_RES).astype('u2')
        )
    CATCHMENT_MASK = tmp.ReadAsArray().astype('u1')
    tmp = None           # flushing!
# #~some.more.PLOTTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     import matplotlib.pyplot as plt
#     plt.imshow(CATCHMENT_MASK, interpolation='none', aspect='equal', origin='upper',
#         cmap='nipy_spectral_r', extent=(llim, rlim, blim, tlim))
#     plt.show()
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# #~IN.CASE.YOU.WANNA.XPORT.THE.MASK.as.ASCII~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     # CORRECT.way (as it registers what is NODATA)
#     tmp = gdal.Rasterize(''
#         , wtrshd.to_json(), xRes=X_RES, yRes=Y_RES, allTouched=True, initValues=-9999., burnValues=1., noData=-9999.
#         , outputType=gdal.GDT_Float32, targetAlignedPixels=True, outputBounds=[llim, blim, rlim, tlim]# , add=0
#         , outputSRS=pp.CRS.from_wkt(WKT_OGC).to_proj4(), format='MEM')
#     # CATCHMENT_MASK = tmp.ReadAsArray()
#     tmv_file = 'tmp-raster_mask.asc'
#     tmv = gdal.GetDriverByName( 'AAIGrid' ).CreateCopy(tmv_file, tmp)
#     tmv = None           # flushing!
#     tmp = None           # flushing!
#     import os
#     os.unlink(f"./{tmv_file.replace('.asc','.prj')}")
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# #~XPORT.IT.as.NUMPY~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     np.save('tmp-raster_mask', CATCHMENT_MASK.astype('u1'), allow_pickle=True, fix_imports=True)
# #~XPORT.IT.as.PICKLE [but don't use it for NUMPYs!]~~~~~~~~~~~~~~~~~~~~~~~~~~#
# # https://stackoverflow.com/a/62883390/5885810
#     import pickle
#     with open('tmp-raster_mask.pkl','wb') as f: pickle.dump(CATCHMENT_MASK.astype('u1'), f)
# # read it back as numpy (through pandas)
#     with open('tmp-raster_mask-buff.pkl', 'rb') as db_file:
#         db_pkl = pickle.load( db_file )
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# DEFINE THE COORDINATES OF THE XY.AXES
    XS, YS = list(map( lambda a,b,c: np.arange(a +c/2, b +c/2, c),
                      [llim,blim], [rlim,tlim], [X_RES,Y_RES] ))
# flip YS??
    YS = np.flipud( YS )      # -> important...so rasters are compatible with numpys

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- SET UP SPACE-TIME DOMAIN & UPDATE PARAMETERS ------------------------ (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- CONSTRUCT THE PDFs (TO SAMPLE FROM) ------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ READ THE CSV.FILE(s) PRODUCED BY THE preprocessing.py SCRIPT ~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def READ_PDF_PAR():
    global PDFS
# read PDF-parameters
# https://stackoverflow.com/a/58227453/5885810  (import tricky CSV)
    PDFS = pd.read_fwf(abspath( join(parent_d, PRE_FILE) ), header=None)
    # PDFS = pd.read_fwf(PRE_FILE, header=None)
    PDFS = PDFS.__getitem__(0).str.split(',', expand=True).set_index(0).astype('f8')


#~ CONSTRUCT PDFs FROM PARAMETERS (stored in 'PDFS') ~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def RETRIEVE_PDF( TAG ):
# TAG: core label/index (in PDFS variable) to construct the pdf on
    subset = PDFS[PDFS.index.str.contains( TAG )].dropna(how='all', axis='columns')

# necessary block as tags COPULA_RHO, MAXINT_PDF, AVGDUR_PDF might have Z-bands
    line = subset.index.str.contains(pat='[Z]\d{1,2}(?!\d)|100')
# https://stackoverflow.com/a/6400969/5885810   # regex for 1-100
# ...in case somebody goes crazy having up to 100 Z-bands!! ('[Z][1-9]' -> otherwise)
    if line.any() == True and Z_CUTS:
        # print('correct!')
        subset = subset[ line ]
        name = np.unique( list(zip( *subset.index.str.split('+') )).__getitem__(1) )
    else:
        subset = subset[ ~line ]
        name = ['']                 # makes "distros" 'universal'

# https://www.geeksforgeeks.org/python-get-first-element-of-each-sublist/
    first = list(list(zip( *subset.index.str.split('+') )).__getitem__(0))
# https://stackoverflow.com/a/6979121/5885810   (numpy argsort equivalent)
# https://stackoverflow.com/a/5252867/5885810
# https://stackoverflow.com/a/46453340/5885810  (difference between strings)
    sort_id = np.unique( list(map( lambda x: x.replace(TAG, ''), first )) )
# the line below makes 1st-PDFs be chosen by default
    sort_id = sort_id[ np.argsort( sort_id.astype('int') )  ]
# # TIP: USE THE LINE BELOW (REPLACING THE LINE ABOVE) IF YOU PREFER 2nd-ids-PDF INSTEAD
# # https://stackoverflow.com/a/16486305/5885810
#     sort_id = sort_id[ np.argsort( sort_id.astype('int') )[::-1]  ]
    group = [subset[subset.index.str.contains( f'{TAG}{i}' )].dropna(
        how='all', axis='columns') for i in sort_id]

    if TAG == 'DATIME_VMF' or TAG == 'DOYEAR_VMF':
# https://cmdlinetips.com/2018/01/5-examples-using-dict-comprehension/
# https://blog.finxter.com/how-to-create-a-dictionary-from-two-numpy-arrays/
        distros = [{A:B for A, B in zip(['p','mus','kappas'],
            [i.to_numpy() for item, i in G.T.iterrows()])} for G in group]
    elif TAG == 'COPULA_RHO':
        distros = [{A:B for A, B in zip(name if Z_CUTS else name,\
            [i.values.ravel().__getitem__(0) for item, i in G.iterrows()])} for G in group]
    else:
        distros = [{A:B for A, B in zip(name,
            [eval(f"stats.{item.split('+').__getitem__(-1)}"\
                  # f"({','.join( i.astype('str').values.ravel() )})")\
                  f"({','.join( i.dropna().astype('str').values.ravel() )})")\
                  for item, i in G.iterrows()] )} for G in group]

    return distros


#~ RETRIEVE THE PDFs & EVALUATE THEIR 'CONSISTENCY' AGAINST #SEASONS ~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def CHECK_PDF():
# https://stackoverflow.com/a/10852003/5885810
# https://stackoverflow.com/q/423379/5885810    (global variables)
    global DATIME, DOYEAR, COPULA, TOTALP, RADIUS, BETPAR, MAXINT, AVGDUR#, Z_CUTS

    try:
        DATIME = RETRIEVE_PDF( 'DATIME_VMF' )
    except IndexError:
        # DATIME = [stats.uniform() for x in range(SEASONS)]
        DATIME = [None for x in range(SEASONS)]
        warnings.warn(f'\nNo DATIME_VMF parameters were found in "{PRE_FILE}".'\
            '\nSTORM3.0 will proceed with TOD (Times Of Day) sampled from a '\
            'UNIFORM distribution. If this is not what you want, please '\
            'accordingly update the aforementioned file.', stacklevel=2)

    try:
        DOYEAR = RETRIEVE_PDF( 'DOYEAR_VMF' )
    except IndexError:
        DOYEAR = RETRIEVE_PDF( 'DOYEAR_PMF' )

    TOTALP = RETRIEVE_PDF( 'TOTALP_PDF' )
    RADIUS = RETRIEVE_PDF( 'RADIUS_PDF' )
    BETPAR = RETRIEVE_PDF( 'BETPAR_PDF' )
    MAXINT = RETRIEVE_PDF( 'MAXINT_PDF' )
    AVGDUR = RETRIEVE_PDF( 'AVGDUR_PDF' )
    COPULA = RETRIEVE_PDF( 'COPULA_RHO' )

# evaluate consistency between lists (lengths must be consistent with #SEASONS)
    test = ['DATIME', 'DOYEAR', 'COPULA', 'TOTALP', 'RADIUS', 'BETPAR', 'MAXINT', 'AVGDUR']
    lens = list(map(len, list(map( eval, test )) ))
# is there are variables having more PDFs than others
    assert len(np.unique(lens)) == 1, 'There are less (defined) PDFs for '\
        f'{" & ".join(np.asarray(test)[np.where(lens==np.unique(lens).__getitem__(0)).__getitem__(0)])} than for '\
        f'{" & ".join(np.delete(np.asarray(test),np.where(lens==np.unique(lens).__getitem__(0)).__getitem__(0)))}.'\
        f'\nPlease modify the file "{PRE_FILE}" (accordingly) to ensure that '\
        'for each of the aforementioned variables exists at least as many PDFs '\
        f'as the number of SEASONS to model ({SEASONS} seasons per year, '\
        'according to your input).'
# if there are more PDFs than the number of seasons (which is not wrong at all)
    if np.unique(lens) > SEASONS:
        warnings.warn(f'\nThe file "{PRE_FILE}" contains parameters for '\
            f'{np.unique(lens).__getitem__(0)} season(s) but you chose to model'\
            f' {SEASONS} season(s) per year.\nSTORM3.0 will proceed using '\
            "PDF-parameters for the season with the lowest 'ID-IndeX' (e.g., "\
            "'TOTALP_PDF1+...', 'RADIUS_PDF1+...', and so forth.)", stacklevel=2)
# if there are more number of seasons than PDFs (STORM duplicates PDFs)
    if np.unique(lens) < SEASONS:
# https://stackoverflow.com/a/45903502/5885810  (replicate list elements)
# https://stackoverflow.com/a/5599313/5885810   (using exec instead of eval)
        for x in test:
            exec( f'{x} = {x}*{SEASONS}' )
        warnings.warn(f'\nThe file "{PRE_FILE}" contains parameters for '\
            f'{np.unique(lens).__getitem__(0)} season(s) but you chose to model'\
            f' {SEASONS} season(s) per year.\nSTORM3.0 will proceed using these'\
            ' parameters for all seasons. If this is not what you want, please'\
            ' update the aforementioned file accordingly.', stacklevel=2)


#~ TOY PDFS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def ALT_CHECK_PDF():
    global DATIME, DOYEAR, COPULA, TOTALP, RADIUS, BETPAR, MAXINT, AVGDUR, WINDIR, WSPEED

    TOTALP = [{'':stats.gumbel_l(5.5116, 0.2262)}, {'':stats.norm(5.3629, 0.3167)}]
    RADIUS = [{'':stats.johnsonsb(1.5187, 1.2696, -0.2789, 20.7977)}, {'':stats.gamma(4.3996, -0.475, 1.399)}]
    BETPAR = [{'':stats.exponnorm(8.2872, 0.0178 ,0.01)}, {'':stats.burr(2.3512, 0.85, -0.0011, 0.0837)}]
    MAXINT = [{'':stats.expon(0.1057 ,6.9955)}, {'':stats.expon(0.1057 ,6.9955)}]
    # [{'Z1': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x1cca438a850>,
    #   'Z2': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x1cca43257c0>,
    #   'Z3': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x1cca4395be0>},
    #  {'Z1': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x1cca438ec10>,
    #   'Z2': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x1cca4325f40>,
    #   'Z3': <scipy.stats._distn_infrastructure.rv_continuous_frozen at 0x1cca438e3d0>}]
    AVGDUR = [{'':stats.geninvgauss(-0.089, 0.77, 2.8432, 82.0786)}, {'':stats.geninvgauss(-0.089, 0.77, 2.8432, 82.0786)}]
    COPULA = [{'':-0.31622}, {'':-0.31622}]
    # [{'Z1': -0.2764573348234358, 'Z2': -0.31246435519305843, 'Z3': -0.44038940293798295},
    #  {'Z1': -0.2764573348234358, 'Z2': -0.31246435519305843, 'Z3': -0.44038940293798295}]
    DATIME = [{'p': np.r_[0.2470, 0.3315, 0.4215], 'mus': np.r_[0.6893, 1.7034, 2.5756],
               'kappas': np.r_[6.418, 3., 0.464]},
              {'p': np.r_[1.], 'mus': np.r_[1.444], 'kappas': np.r_[1.0543]}]
    DOYEAR = [{'p': np.r_[0.054, 0.089, 0.07, 0.087, 0.700], 'mus': np.r_[1.9, 0.228, 1.55, 1.172, 0.558] -np.pi /2,
               'kappas': np.r_[105.32,  51.97,  87.19,  52.91, 6.82]},
              {'p': np.r_[1.], 'mus': np.r_[0.7136], 'kappas': np.r_[3.9]}]
    # a KAPPA~=0 tranforms a VM into a UNIFORM.DISTRO from [-pi,pi]
    # https://rstudio.github.io/tfprobability/reference/tfd_von_mises.html
    WINDIR = [{'p': np.r_[1.], 'mus': np.r_[np.pi *.9], 'kappas': np.r_[0.00001]},
              {'p': np.r_[1.], 'mus': np.r_[np.pi *.1], 'kappas': np.r_[0.00001]}]
# until you parameterize WIN.DIRECTION use scypi's VONMISES [it's ~100x faster than VMM-package!]
    WINDIR = [{'':stats.vonmises(.9 *np.pi, .00001)}, {'':stats.vonmises(.1 *np.pi, .00001)}]
    WSPEED = [{'':stats.norm(7.55, 1.9)}, {'':stats.norm(7.55, 1.9)}]

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- CONSTRUCT THE PDFs (TO SAMPLE FROM) --------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RANDOM SMAPLING --------------------------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ N-RANDOM SAMPLES FROM 'ANY' GIVEN PDF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def RANDOM_SAMPLING( PDF, N ):
# PDF: scipy distribution_infrastructure (constructed PDF)
# N  : number of (desired) random samples
    xample = PDF.rvs( size=N )
    # # for reproducibility
    # xample = PDF.rvs( size=N, random_state=npr.RandomState(npr.Philox(12345)) )
    # xample = PDF.rvs( size=N, random_state=npr.RandomState(npr.PCG64DXSM(1337)) )
    # xample = PDF.ppf( npr.RandomState(777).random( N ) ) # -> TOTALP for Matlab 'compatibility'
    return xample


#~ TRUNCATED N-RANDOM SAMPLES FROM 'ANY' GIVEN PDF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def TRUNCATED_SAMPLING( PDF, LIMITS, N ):
# LIMITS: PDF boundaries to constrain the sampling
# find the CDF for the limit (if any limit at all)
    LIMITS = [PDF.cdf(x) if x else None for x in LIMITS]
# if there are None, replace it by the lowest/highest possible CDF(s)
    if None in LIMITS:
# https://stackoverflow.com/a/50049044/5885810  (None to NaN to Zero)
        LIMITS = np.nan_to_num( np.array(LIMITS, dtype='f8') ) + np.r_[0, 1]
    xample = PDF.ppf( npr.uniform(LIMITS[0], LIMITS[-1], N) )
    # # for reproducibility
    # xample = PDF.ppf( npr.RandomState(npr.SFC64(54321)).uniform(LIMITS[0], LIMITS[-1], N) )   # -> RADIUS
    # xample = PDF.ppf( npr.RandomState(npr.PCG64(2001)).uniform(LIMITS[0], LIMITS[-1], N) )    # -> BETPAR
    # xample = PDF.ppf( npr.RandomState(555).uniform(LIMITS[0], LIMITS[-1], N) ) # -> RADIUS for Matlab 'compatibility'
    # xample = PDF.ppf( npr.RandomState(999).uniform(LIMITS[0], LIMITS[-1], N) ) # -> BETPAR for Matlab 'compatibility'
    return xample


#~ RETRIEVE TOTAL SEASONAL/MONSOONAL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def SEASONAL_RAIN( PDF, seas, BAND='', N=1 ):
# sample N values of TOTALP & transform them from ln-space
    total = np.exp( RANDOM_SAMPLING( PDF[ seas ][ BAND ], N ) )
    return total


#~ INDEPENDENT & ALTERED SAMPLING OF TOTAL SEASONAL/MONSOONAL ~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def ENHANCE_SR( seas, TOTALP_DISTRO ):# TOTALP_DISTRO=TOTALP

#~IF SAMPLING FROM TOTALP~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# sample total monsoonal rainfall to reach & increase/decrease (such) total monsoonal rainfall
    # seas_rain = SEASONAL_RAIN( TOTALP_DISTRO, seas ) * (1 + PTOT_SC[ seas ] + ((simy +0) *PTOT_SF[ seas ]))
    seas_rain = SEASONAL_RAIN( TOTALP_DISTRO, seas ) * 1
    # seas_help = SEASONAL_RAIN( TOTALP_DISTRO, seas )
    # seas_rain = ( seas_help * (1 + PTOT_SC[ seas ]) ) * (1 + ((simy +1) *PTOT_SF[ seas ]))
    return seas_rain #/100


# #~ PERFORM A POISSON.POINT.PROCESS (INSIDE A SHP) [SLOW approach]~~~~~~~~~~~~~~#
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# def SCENTRES( NUM_S ):
# """
# if using this methodology, please activate/import:
#     import libpysal as ps
#     from pointpats import PoissonPointProcess, Window, PointPattern
# """
# # FROM: https://nbviewer.org/github/pysal/pointpats/blob/main/notebooks/process.ipynb#Random-Patterns
# """
# This methodolody (of this function) could potentially be implemented IF
# some 'Clustered Point Pattern' might be looked for (e.g., regionalization??).
# Its main drawback is that it is really slow.
# Thus (and in the meantime) it's very advisable to follow its alternative "pran.poisson(...".
# As of "libpysal 4.7.0" (02/06/2023), it comes also with a default warning:
# https://pysal.org/libpysal/_modules/libpysal/cg/shapes.html
#     ""
#     lib\site-packages\libpysal\cg\shapes.py:103: FutureWarning:
#         Objects based on the `Geometry` class will deprecated and removed in a future version of libpysal.
#         warnings.warn(dep_msg, FutureWarning)
#     ""
# The methodology below works with the '.geometry' method (from geoPandas);
# nevertheless, if some future complications arise (from this '.geometry' method),
# we also present alternatives to construct the SHP from vertices.
# You can remove the warning by following (e.g.):
#     import warnings
#     warnings.filterwarnings("ignore", message=".*`Geometry` class will deprecated and removed in a future version of libpysal*")
# """

# # 1. transform SHAPELY into PYSAL
# # ... don't know exactly how to proceed it the SHP has VOIDS/HOLES
#     sal_shp = ps.cg.asShape( BUFFRX.geometry.xs(0) )
#     # # OR (a couple of alternatives yielding the same result??)...
#     # sal_shp = ps.cg.shapes.asShape( BUFFRX.geometry.xs(0) )
#     # sal_shp = ps.cg.shapely_ext.boundary( BUFFRX.geometry.xs(0) )  # too.slow + warning

#     # # OR... one can create a PYSAL.POLYGON if knowing the VERTICES
#     # xs, ys = BUFFRX.geometry.xs(0).exterior.coords.xy
#     # vertixs = list(zip( xs, ys ))
#     # # OR (same as above but ONE.LINER)...
#     # # https://stackoverflow.com/a/7558990/5885810     (packing as tuples/list)
#     # vertixs = list(zip( *BUFFRX.geometry.xs(0).exterior.coords.xy ))
#     # sal_shp = ps.cg.Polygon( list(zip( *BUFFRX.geometry.xs(0).exterior.coords.xy )) )

# # 2. transform the PYSAL into a WINDOW
# # ... '.PARTS' (along with '.VERTICES') seems to give you the vertices (so potentially you can bypass 1.?)
#     wndw_bffr = Window( sal_shp.parts )
#     # # example on 'WINDOW' if one decides to bypass PYSAL!
#     # # https://stackoverflow.com/a/34551914/5885810    (PANDAS to TUPLEs)
#     # wndw_bffr = Window( list(BUFFRX.get_coordinates().itertuples(index=False)) )
#     # # # this one (in.pple) tells PYSAL IT's ONE.WHOLE 'part' (but seems to equally work as the above line)
#     # # wndw_bffr = Window( [list(BUFFRX.get_coordinates().itertuples(index=False))] )

# # 3. do the spatial.random.sampling (inside the SHP)
# # simulate a Complete Spaital Randomness (CSR) process inside a SHP (NUM_S==n-points, 1==1-realization)
# # ... (maybe it'd be a good idea to compute many realizations, and then sample from them iteratively)??
# # "asPP"==True  -> generates a point pattern (pandas that can be easily visualized)
# # "asPP"==False -> generates a point series  (just a numpy)
# # "conditioning"==True  -> simulates a lamda-conditioned CSR  (variable NUM_S)
# # "conditioning"==False -> simulates a N-conditioned CSR      (exactly  NUM_S)

# # let's use (only) "conditioning"==True
# # !!the more REALIZATIONS and more POINTS, the slower it gets!!

# # #~IF POINT.PATTERN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     # %timeit
#     # samples = PoissonPointProcess(wndw_bffr, NUM_S, 1, conditioning=True,  asPP=True) # gives.warning
#     # # 2.49 s ± 199 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
#     # # 23.4 s ± 1.24 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
#     # # 4min 10s ± 4.07 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
#     # # wanna visualize the points?
#     # samples.realizations[0].plot(window=True)
#     # # return storm.centres as NUMPY
#     # CENTS = samples.realizations[0].points.to_numpy() # if only.1.realization computed

# # #~IF POINT.SERIES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     # %timeit
#     samples = PoissonPointProcess(wndw_bffr, NUM_S, 1, conditioning=True,  asPP=False) # gives.warning
#     # 2.53 s ± 127 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
#     # # wanna visualize the points? (there's no SHP here!)
#     # PointPattern( samples.realizations[0] ).plot(window=True, hull=True, title='point series')
#     # return storm.centres as NUMPY
#     CENTS = samples.realizations[0]

#     return CENTS


#~ PERFORM A POISSON.POINT.PROCESS (INSIDE A SHP) [FAST approach]~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def SCENTRES( AREA_SHP, NUM_S ):# AREA_SHP=REGIONS['mask'][0]
# FROM: https://stackoverflow.com/a/69630606/5885810    (random poitns in SHP)
    """
This is the chosen alternative (also coming from libpysal) as it's very fast.
It potentially also allows for some 'Clustered Point Pattern' generation,
throughout the methods "pran.cluster_poisson(...", for instance.

As of "libpysal 4.7.0", "pointpats 2.3.0", and "numba 0.57.0" (02/06/2023),
it comes with warnings:
https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit
    ""
    lib\site-packages\libpysal\cg\alpha_shapes.py:261: NumbaDeprecationWarning:
        The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator.
        The implicit default value for this argument is currently False, but it will
        be changed to True in Numba 0.59.0.  See [LINK ABOVE!] for details.
    ""
We don't know the future implications of this warning (when having, e.g., numba 0.59).
As of (02/06/2023), this methodology works... and in the eventual case of a future
breakdown, we have also proposed here an alternative "SCENTRES", circunventing the
use of "pran.poisson(..." (please check its drawbacks though).
You can remove the warning by following (e.g.):
    import warnings
    warnings.filterwarnings('ignore', message=".*The 'nopython' keyword.*")
# OR
    from numba.core.errors import NumbaDeprecationWarning
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    """

    # %timeit
    CENTS = pran.poisson( AREA_SHP.geometry.xs(0), size=NUM_S )
    # 104 ms ± 2.34 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

    # # you wanna PLOT the STORM CENTRES??
    # import matplotlib.pyplot as plt
    # import cartopy.crs as ccrs
    # fig = plt.figure(figsize=(10,10), dpi=300)
    # # ax = plt.axes(projection=ccrs.epsg(42106)) # this EPSG code doesn't officially exist!
    # ax = plt.axes( )
    # ax.set_aspect(aspect='equal')
    # for spine in ax.spines.values(): spine.set_edgecolor(None)
    # fig.tight_layout(pad=0)
    # AREA_SHP.plot(edgecolor='xkcd:amethyst', alpha=1., zorder=2, linewidth=.77,
    #               ls='dashed', facecolor='None', ax=ax)
    # plt.scatter(CENTS[:,0], CENTS[:,1], marker='P', s=37, edgecolors='none')
    # plt.show()

    return CENTS


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
        # # to check out if the sampling is done correctly
        # plt.hist(doys, bins=365)
# into actual dates
        dates = list(map(lambda d:
            datetime(year=DATE_POOL[ seas ][0].year,month=1,day=1) +\
            relativedelta(yearday=int( d )), doys.round(0) ))
        sates = pd.Series( dates )              # to pandas
# chopping into limits
        sates = sates[(sates>=DATE_POOL[ seas ][ 0]) & (sates<=DATE_POOL[ seas ][-1])]
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

    # # taken from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises.html
    # fig = plt.figure(figsize=(12, 6))
    # left = plt.subplot(121)
    # right = plt.subplot(122, projection='polar')
    # x = np.linspace(-np.pi, np.pi, 500)
    # # # lines below computes the pdf.density (given the parameters)
    # # vonmises_pdf = stats.vonmises.pdf(loc, kappa, x)  # using SCIPY's VONMISES
    # # # the line below is for plotting vonmises from VONMISESMIXTURES (package)
    # vonmises_pdf = ( PLTVAR[ seas ]['p'] *vonmises.density(x, PLTVAR[ seas ]['mus'], PLTVAR[ seas ]['kappas']) ).sum(axis=1)
    # ticks = [0, 0.15, 0.3]
    # # left
    # left.plot(x, vonmises_pdf)
    # left.set_yticks(ticks)
    # number_of_bins = int(np.sqrt(number_of_samples))
    # left.hist(samples, density=True, bins=number_of_bins)
    # left.set_title("Cartesian plot")
    # left.set_xlim(-np.pi, np.pi)
    # left.grid(True)
    # # right
    # right.plot(x, vonmises_pdf, label="PDF")
    # right.set_yticks(ticks)
    # right.hist(samples, density=True, bins=number_of_bins, label="Histogram")
    # right.set_title("Polar plot")
    # right.legend(bbox_to_anchor=(0.15, 1.06))
    # # xport
    # plt.savefig(f'tmp_uniform_vmX.png', bbox_inches='tight',pad_inches=0.02, facecolor=fig.get_facecolor())
    # plt.close()
    # plt.clf()


#~ SAMPLE DAYS.OF.YEAR and TIMES.OF.DAY (DISCRETE approach) ~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def TOD_DISCRETE( N, seas, simy ):# N=120
    M = N
    all_dates = []
    while M>0:
        soys = RANDOM_SAMPLING( DOYEAR[ seas ][''], M )
# chopping into limits
        doys = soys[(soys>=DATE_POOL[ seas ].__getitem__(0).timetuple().tm_yday) &\
                    (soys<=DATE_POOL[ seas ].__getitem__(1).timetuple().tm_yday)]
        # plt.hist(doys, bins=365)
# into actual dates
        dates = list(map(lambda d:
            datetime(year=DATE_POOL[ seas ].__getitem__(0).year,month=1,day=1) +\
            relativedelta(yearday=d), doys ))
        sates = pd.Series( dates )              # to pandas
        M = len(soys) - len(sates)
        # print(M)
# updating to SIMY year (& storing)
        # all_dates.append( sates + pd.DateOffset(years=simy) )
        all_dates.append( sates.map(lambda d:d +relativedelta(years=simy)) )
        # # the line above DOES NOT give you errors when dealing with VOID arrays
    all_dates = pd.concat( all_dates, ignore_index=True )

    """
If you're unlucky to be stuck with "DISCRETE"...
then there's no point in using circular on TOD, is it?'
    """
# TIMES
# sampling from a NORMAL distribution
    times = npr.uniform(0, 1, N) *24
    # # to check out if the sampling is done correctly
    # plt.hist(times, bins=24)
# SECONDS since DATE_ORIGIN
# https://stackoverflow.com/a/50062101/5885810
# https://stackoverflow.com/a/67105429/5885810  (chopping milliseconds)
    stamps = np.asarray( list(map(lambda d,t:
        ((d + timedelta(hours=t)) - DATE_ORIGEN).total_seconds(),
        all_dates.dt.tz_localize(TIME_ZONE), times)) )
# # pasting and formatting
# # https://stackoverflow.com/a/67105429/5885810  (chopping milliseconds)
#     stamps = np.asarray(list(map(lambda d,t: (d + timedelta(hours=t)).isoformat(timespec='seconds'), dates, times)))
    stamps = np.round(stamps, 0)#.astype('u8')
# STAMPS here are scaled to the output.TIMENC.resolution
    return stamps *TIME_DICT_[ TIME_OUTNC ]

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RANDOM SMAPLING ----------------------------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RASTER MANIPULATION ----------------------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ CREATE AN OUTER RING/POLYGON ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# *1e3 to go from km to m
def LAST_RING( all_radii, CENTS ):#, MINRADIUS ):# all_radii=RADII
# "resolution" is the number of segments in which a.quarter.of.a.circle is divided into.
# ...now it depends on the RADII/RES; the larger a circle is the more resolution it has.
    ring_last = list(map(lambda c,r: gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=[c[0]], y=[c[1]] ).buffer( r *1e3,
            # resolution=int((3 if r < 1 else 2)**np.ceil(r /2)) ),
            # resolution=np.ceil(r /MINRADIUS) +1 ), # or maybe... "+1"??
            resolution=np.ceil(r /MINRADIUS) +2 ), # or maybe... "+2"??
        # crs=f'EPSG:{WGEPSG}'), CENTS, all_radii))
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
        # crs=f'EPSG:{WGEPSG}') , r, p)) ), CENTS, all_radii, all_rain))
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
        # test = zonal_stats(abspath( join(parent_d, SHP_FILE) ), './data_WG/dem/WGdem_wgs84.tif', stats='count min mean max median')
        # IF YOUR DEM IS IN WGS84... RE-PROJECT THE POLYGONS TO 4326 (WGS84)
        ztats = zonal_stats(vectors=Z_OUT.to_crs(epsg=4326).geometry, raster=DEM_FILE, stats=Z_STAT)
        # # OTHERWISE, A FASTER APPROACH IS HAVING THE DEM/RASTER IN THE LOCAL CRS
        # # ...i.e., DEM_FILE=='./data_WG/dem/WGdem_26912.tif'
        # ztats = zonal_stats(vectors=Z_OUT.geometry, raster=DEM_FILE, stats=Z_STAT)
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

#~ ROUND TIME.STAMPS to the 'T_RES' FLOOR! (or NEAREST 'T_RES') ~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def BASE_ROUND( stamps, base= T_RES):
# https://stackoverflow.com/a/2272174/5885810
# "*60" because STAMPS come in seconds; and T_RES is in minutes
    base = base *TIME_DICT_[ TIME_OUTNC ] *60
    iround = (base *(np.ceil(stamps /base) -1))#.astype( TIMEINT )
# # activate line below if you want to the NEAREST 'T_RES' (instead of FLOOR)
#     iround = (base *(stamps /base).round())#.astype( TIMEINT )
    return iround


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


#~ [TIME BLOCK] SAMPLE DATE.TIMES and XPAND THEM ACCORDING MOVING VECTORS ~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def QUANTIZE_TIME( NUM_S, seas, simy, durations ):# durations=DUR_S[ okdur ]
    # global i_scaling
# sample some dates (to capture intra-seasonality & for NC.storing)
    DATES = eval(f'{tod_fun}( {NUM_S}, {seas}, {simy} )')
    # DATES = TOD_CIRCULAR( NUM_S, seas, simy )
# round starting.dates to nearest.floor T_RES
    RATES = BASE_ROUND( DATES )
# turn the DUR_S into discrete time.slices
    i_scaling = TIME_SLICE( (DATES+ -1*RATES), durations )
# xploding of discrete timestamps (per storm.cluster)
    MATES = np.concatenate( list(map(lambda r_s,i_s:
        # np.arange(start=r_s, stop=r_s + 60*T_RES*len(i_s), step=60*T_RES),
        np.arange(start=r_s, stop=r_s + (T_RES *TIME_DICT_[ TIME_OUTNC ] *60)*len(i_s), step=T_RES *TIME_DICT_[ TIME_OUTNC ] *60),
        RATES, i_scaling)) ).astype( TIMEINT )
    return MATES, i_scaling


# #~ ONE 1ST APPROACH USED TO DISCRETIZE TIME [NOT USED ANYMORE!] ~~~~~~~~~~~~~~~#
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# def SPACETIME( S_M, i_s, r_s ):# S_M=STORM_MATRIX[23]; i_s=i_scaling[23]; r_s=RATES[23]
#     # https://stackoverflow.com/a/32171971/5885810  -> replicate a 2D.numpy (along a 3D)
#     one_s = np.repeat(S_M[np.newaxis, :, :], len(i_s), axis=0)
#     # np.max(one_s, axis=(1,2))  *i_s
# # scale.down rainfall intensity into rainfall depths (for a given T_RES)
#     # https://stackoverflow.com/q/14513222/5885810  -> multiply np of diff.dimensions
#     one_s = one_s * i_s[:, None, None]
# # convert into Xarray
#     one_s = xr.DataArray(data=one_s, dims=['time','row','col'])
#     one_s.coords['time'] = np.arange(start=r_s, stop=r_s + 60*T_RES*len(i_s), step=60*T_RES).astype('u8')
#     return one_s

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- MISCELLANEOUS TO TIME-DISCRETIZATION -------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- EXTRA-CHUNK OF MISCELLANEOUS FUNCTIONS ---------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# #~ REMOVE (ACCORDINGLY) DURATIONS OUTSIDE [MIN_DUR, MAX_DUR].range ~~~~~~~~~~~~#
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# def CHOP( DUR_S, DATES, STORM_MATRIX ):
# # find the indexes outside the limits
#     outdur = np.concatenate((np.where(DUR_S<MIN_DUR).__getitem__(0) if MIN_DUR else np.empty(0),
#         np.where(DUR_S>MAX_DUR).__getitem__(0) if MAX_DUR else np.empty(0))).astype('int')
#     d_bool = ~np.in1d(range(len(STORM_MATRIX)), outdur)
# # update 'vectors'
#     return DUR_S[ d_bool ], DATES[ d_bool ], [item for i, item in enumerate(STORM_MATRIX) if d_bool[i]]

#~ INDEX ALL DURATIONS OUTSIDE [MIN_DUR, MAX_DUR].range ~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def CHOP( DUR_S ):
# find the indexes outside the limits
    outdur = np.concatenate((np.where(DUR_S<MIN_DUR).__getitem__(0) if MIN_DUR else np.empty(0),
        np.where(DUR_S>MAX_DUR).__getitem__(0) if MAX_DUR else np.empty(0))).astype('int')
    d_bool = ~np.in1d(range(len(DUR_S)), outdur)
# update 'vectors'
    return d_bool


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
    # stride = list(map( np.cumsum, list(map(np.multiply, wspeed *1.9, list(map(lambda x:np.r_[0,x[:-1]], i_scaling)) )) ))
    stride = list(map( np.cumsum,
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


#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- NC.FILE CREATION -------------------------------------------------- (START) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~ CUSTOM.ROUNDING TO SPECIFIC.BASE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# # https://stackoverflow.com/a/18666678/5885810
def ROUNDX(x, prec=3, base=PRECISION):
    return (base * (np.array(x) / base).round()).round(prec)


#~ TO SCALE 'DOWN' FLOATS TO INTEGERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def NCBYTES_RAIN():
    global SCL, ADD, MINIMUM, VOIDXR

    INTEGER = int(RAINFMT[-1])
# # if one wants signed integers (e.g.)...
#     MINIMUM = -(2**(INTEGER *8 -1))       # -->     -32768  (smallest  signed integer of 16 Bits)
#     MAXIMUM = +(2**(INTEGER *8 -1)-1)     # -->     +32767  (largest   signed integer of 16 Bits)
# # if one wants un-signed integers (e.g.)...
    # MINIMUM = 0                             # -->          0  (0 because it's unsigned)
    MINIMUM = 1                             # -->          1  (1 because i need 0 for filling)
    MAXIMUM = +(2**( INTEGER *8 )) -1       # -->      65535  (largest unsigned integer of 16 Bits -> 0 also counts)
    # MAXIMUM = +(2**( 32 )) -1             # --> 4294967296  (largest unsigned integer of 32 Bits)

# # run your own (customized) tests
# temp = 3.14159
# seed = 1133
# epsn = 0.0020
# while temp > epsn:
#     temp = (seed - 0.) / (MAXIMUM - (MINIMUM + 1))
#     seed = seed - 1
# print( (f'starting from 0, you\'d need a max. of: {seed+1} to guarantee an epsilon of {epsn}') )
# # starting from 0, you'd need a max. of:  65 to guarantee an epsilon of 0.001
# # starting from 0, you'd need a max. of: 131 to guarantee an epsilon of 0.002
# # starting from 0, you'd need a max. of: 196 to guarantee an epsilon of 0.003
# # starting from 0, you'd need a max. of: 262 to guarantee an epsilon of 0.004
# # starting from 0, you'd need a max. of: 327 to guarantee an epsilon of 0.005
# # starting from 0, you'd need a max. of: 393 to guarantee an epsilon of 0.006
# # starting from 0, you'd need a max. of: 655 to guarantee an epsilon of 0.01
# # starting from 0, you'd need a max. of: 429496 to guarantee an epsilon of 0.0001 (for INTEGER==4)

# NORMALIZING THE RAINFALL SO IT CAN BE STORED AS 16-BIT INTEGER (65,536 -> unsigned)
# https://stackoverflow.com/a/59193141/5885810      (scaling 'integers')
# https://stats.stackexchange.com/a/70808/354951    (normalize data 0-1)
    iMIN = 0.
# if you want a larger precision (or your variable is in the 'low' scale,
# ...say Summer Temperatures in Celsius) you must/could lower this limit.
    iMAX = (MAXIMUM -MINIMUM) *PRECISION
    # 131.070 -> for MINIMUM==0
    # 131.068 -> for MINIMUM==1
    SCL = (iMAX - iMIN) / (MAXIMUM -MINIMUM)# -1)   # if one wants UNsigned INTs
    # SCL = (iMAX - iMIN) / (MAXIMUM - (MINIMUM + 0))
    # ADD = iMAX - SCL * (MAXIMUM -0)
    ADD = iMAX - SCL * MAXIMUM

    # # testing
    # allv = (np.linspace(MINIMUM +1, MAXIMUM, MAXIMUM) -1) *PRECISION
    # allv.shape
    # # Out[22]: (65535,)
    # allv[-1] == iMAX
    # # Out[22]: True
    # allt = ((allv - ADD) /SCL).round(0).astype('u2')
    # np.where(np.diff(allt)!=1)
    # Out[22]: (array([], dtype=int64),)
    # vall = (allt *SCL) +ADD
    # vall = ROUNDX( (allt *SCL) +ADD )

    if RAINFMT[0]=='f':
        SCL = 1.
        ADD = 0.
        MINIMUM = 0.

    VOIDXR = np.empty((0, len(YS), len(XS))).astype( f'{RAINFMT}' )
    # VOIDXR.fill( np.round((0 - ADD) / SCL, 0).astype( f'{RAINFMT}' ) )
    VOIDXR.fill( np.round(0, 0).astype( f'{RAINFMT}' ) )


#~ SKELETON OF THE NC (OUPUT) FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def NC_FILE_I( nc, nsim ):
    global tag_y, tag_x, sref_name, maskxx
# define SUB.GROUP and its dimensions
    sub_grp = nc.createGroup(f'run_{"{:02d}".format(nsim+1)}')

    sub_grp.createDimension('y', len(YS))
    sub_grp.createDimension('x', len(XS))
    # sub_grp.createDimension('t', None)                  # unlimited
    sub_grp.createDimension('n', NUMSIMYRS)

#- LOCAL.CRS (netcdf definition) -----------------------------------------------
#-------------------------------------------------------------------------------
    """
Customization of these parameters for your local CRS is relatively easy!.
All you have to do is to 'convert' the PROJ4 (string) parameters of your (local)
projection into CF conventions.
The following links offer you a guide on how to do so,
and the conventions between CF & PROJ4 & WKT:
# https://cfconventions.org/wkt-proj-4.html
# http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#appendix-grid-mappings
# https://spatialreference.org/
In this case, the PROJ4.string (from the WKT) is:
    **[pp.CRS.from_wkt(WKT_OGC).to_proj4() or pp.CRS.from_epsg(WGEPSG).to_proj4()]**
    '+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +ellps=sphere +units=m +no_defs +type=crs'
which is very similar to the one provided by ISRIC/andresQuichimbo:
    '+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
The amount (and type) of parameters will vary depending on your local CRS.
For instance [http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#lambert-azimuthal-equal-area],
    the LAEA (lambert_azimuthal_equal_area) system requieres 4 parameters:
    longitude_of_projection_origin, latitude_of_projection_origin, false_easting, false_northing
    which correspond to PROJ4 parameters [https://cfconventions.org/wkt-proj-4.html]:
    +lon_0, +lat_0, +x_0, +y_0
    ***
The use of PROJ4 is now being discouraged (https://inbo.github.io/tutorials/tutorials/spatial_crs_coding/);
neverthelesss, and for now, it still works under this framework to store data
in the local CRS, and at the same time be able to visualize it in WGS84
(via, e.g., https://www.giss.nasa.gov/tools/panoply/) without the need to
transform (and store) local coordinates into Lat-Lon.

[02/08/23] We now use RIOXARRAY to "attach" the CRS, and establish some common
parameters to generate some consistency when reading future? random rain-fields.

IF FOR SOME REASON YOU'D ULTIMATELY PREFER TO STORE YOUR DATA IN WGS84, COMMENT
OUT ALL THIS SECTION & ACTIVATE THE SECTION "- WGS84.CRS (netcdf definition) -"
    """
    # create an empty-and-local xarray
    # xoid = EMPTY_MAP2()
    xoid = EMPTY_MAP( YS, XS, WKT_OGC )
    sref_name = 'spatial_ref'

    # grid = sub_grp.createVariable(sref_name, 'int')     # 'int'=='i4'
    grid = sub_grp.createVariable(sref_name, 'u1')
    grid.long_name = sref_name
    # xoid.rio.grid_mapping   # assuming == 'spatial_ref'
    grid.crs_wkt = xoid.spatial_ref.attrs['crs_wkt']
    grid.spatial_ref = xoid.spatial_ref.attrs['crs_wkt']
# https://www.simplilearn.com/tutorials/python-tutorial/list-to-string-in-python
# https://www.geeksforgeeks.org/how-to-delete-last-n-rows-from-numpy-array/
    # grid.GeoTransform = ' '.join( map(str, list(xoid.rio.transform()) ) )
    # # this is apparently the "correct" way to store the GEOTRANSFORM!
    grid.GeoTransform = ' '.join( map(str, np.roll(np.asarray(xoid.rio.transform()).reshape(3,3),
        shift=1, axis=1)[:-1].ravel().tolist() ))                   # [:-1] removes last row
    # ~~~ from CFCONVENTIONS.ORG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ [start]
    grid.grid_mapping_name = 'lambert_azimuthal_equal_area'
    grid.latitude_of_projection_origin = 5
    grid.longitude_of_projection_origin = 20
    grid.false_easting = 0
    grid.false_northing = 0
    # grid.horizontal_datum_name = 'WGS84'                          # in pple this can also be un-commented!
    grid.reference_ellipsoid_name = 'sphere'
    grid.projected_crs_name = 'WGS84_/_Lambert_Azim_Mozambique'     # new in CF-1.7 [https://cfconventions.org/wkt-proj-4.html]
    # ~~~ from CFCONVENTIONS.ORG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ [end]
    # ~~~ from https://publicwiki.deltares.nl/display/NETCDF/Coordinates [start]
    grid._CoordinateTransformType = 'Projection'
    grid._CoordinateAxisTypes = 'GeoY GeoX'

    # # STORING LOCAL COORDINATES
    yy = sub_grp.createVariable('projection_y_coordinate', 'i4', dimensions=('y')
                                ,chunksizes=CHUNK_3D( [ len(YS) ], valSize=4))
    xx = sub_grp.createVariable('projection_x_coordinate', 'i4', dimensions=('x')
                                ,chunksizes=CHUNK_3D( [ len(XS) ], valSize=4))
    yy[:] = YS
    xx[:] = XS
    # yy[:] = np.flipud( np.linspace(3498500,3520500,23) )
    # xx[:] = np.linspace(575500,611500,37)
    yy.coordinates = 'projection_y_coordinate'
    xx.coordinates = 'projection_x_coordinate'
    yy.units = 'meter'
    xx.units = 'meter'
    yy.long_name = 'y coordinate of projection'
    xx.long_name = 'x coordinate of projection'
    yy._CoordinateAxisType = 'GeoY'
    xx._CoordinateAxisType = 'GeoX'
    yy.grid_mapping = sref_name
    xx.grid_mapping = sref_name

# #- WGS84.CRS (netcdf definition) ---------------------------------------------
# #-----------------------------------------------------------------------------
#     # http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#_trajectories
#     grid = sub_grp.createVariable(sref_name, 'int')
#     grid.long_name = sref_name
#     # ~~~ RIO.XARRAY defaults for WGS84 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ [start]
# # # one alternative (once the module RCRS is properly called)
# #     grid.crs_wkt = rcrs.CRS.from_epsg(4326).to_wkt()
# #     grid.spatial_ref = rcrs.CRS.from_epsg(4326).to_wkt()
# # ...or maybe one that is more wholesome?
#     grid.crs_wkt = pp.crs.CRS(4326).to_wkt()
#     grid.spatial_ref = pp.crs.CRS(4326).to_wkt()
# # the line below ONLY works IF "XOID" was generated in WGS84! (... or reprojected to it!)
#     grid.GeoTransform = ' '.join( map(str, list(xoid.rio.transform()) ) )
#     grid.grid_mapping_name = 'latitude_longitude'
#     grid.semi_major_axis = 6378137.
#     grid.semi_minor_axis = 6356752.314245179
#     grid.inverse_flattening = 298.257223563
#     grid.reference_ellipsoid_name = 'WGS 84'
#     grid.longitude_of_prime_meridian = 0.
#     grid.prime_meridian_name = 'Greenwich'
#     grid.geographic_crs_name = 'WGS 84'
#     # ~~~ RIO.XARRAY defaults for WGS84 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ [end]
#     # ~~~ from https://publicwiki.deltares.nl/display/NETCDF/Coordinates [start]
#     grid._CoordinateAxisTypes = 'Lat Lon'

#     # STORING WGS84 COORDINATES
#     lat, lon = pp.Transformer.from_proj(pp.CRS.from_wkt(WKT_OGC).to_proj4(),'EPSG:4326').transform(
#         np.meshgrid(XS,YS).__getitem__(0), np.meshgrid(XS,YS).__getitem__(-1),
#         zz=None, radians=False)
#     # https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1
#     yy = sub_grp.createVariable('latitude' , 'f8', dimensions=('y','x')
#                                 ,chunksizes=CHUNK_3D( [ len(YS), len(XS) ], valSize=8))
#     xx = sub_grp.createVariable('longitude', 'f8', dimensions=('y','x')
#                                 ,chunksizes=CHUNK_3D( [ len(YS), len(XS) ], valSize=8))
#     yy[:] = lat
#     xx[:] = lon
#     yy.coordinates = 'latitude'
#     xx.coordinates = 'longitude'
#     yy.units = 'degrees_north'
#     xx.units = 'degrees_east'
#     yy.long_name = 'latitude coordinate'
#     xx.long_name = 'longitude coordinate'
#     yy._CoordinateAxisType = 'Lat'
#     xx._CoordinateAxisType = 'Lon'
#     yy.grid_mapping = sref_name
#     xx.grid_mapping = sref_name
# #-----------------------------------------------------------------------------

    tag_y = yy.getncattr('coordinates')
    tag_x = xx.getncattr('coordinates')

# store the MASK
    mask_chunk = CHUNK_3D( [ len(YS), len(XS) ], valSize=1)
    ncmask = sub_grp.createVariable('mask', 'i1', dimensions=('y','x')
        ,chunksizes=mask_chunk, zlib=True, complevel=9)#,fill_value=0
    ncmask[:] = CATCHMENT_MASK.astype('i1')
    ncmask.grid_mapping = sref_name
    ncmask.long_name = 'catchment mask'
    ncmask.description = '1 means catchment or region : 0 is void'
    #ncmask.coordinates = f'{yy.getncattr("coordinates")} {xx.getncattr("coordinates")}'
    ncmask.coordinates = f'{tag_y} {tag_x}'

# # if ANOTHER/OTHER variable is needed
#     ncxtra = sub_grp.createVariable('duration', 'f4', dimensions=('t','n')
#         ,zlib=True, complevel=9, fill_value=np.nan)# ,fill_value=np.r_[0].astype('u2'))
#     ncxtra.long_name = 'storm duration'
#     ncxtra.units = 'minutes'
#     ncxtra.precision = f'{1/60}'                        # (1 sec); see last line of 'NC_FILE_II'
#     ncxtra.grid_mapping = sref_name
#     # ncxtra.scale_factor = dur_SCL
#     # ncxtra.add_offset = dur_ADD
#     iixtra = sub_grp.createVariable('sampled_total', 'f4', dimensions=('n')
#         ,zlib=True, complevel=9, fill_value=np.nan)
#     iixtra.long_name = 'seasonal total from PDF'
#     iixtra.units = 'mm'
#     iixtra.grid_mapping = sref_name

    maskxx = sub_grp.createVariable('k_means', datatype='i1', dimensions=('y','x')
        ,chunksizes=mask_chunk, zlib=True, complevel=9,
        fill_value=np.array( -1 ).astype('i1'))#,least_significant_digit=3)
    maskxx.grid_mapping = sref_name
    maskxx.long_name = 'k-means clusters'
    maskxx.description = '-1 indicates region out of any cluster'
    maskxx.coordinates = f'{tag_y} {tag_x}'

    return sub_grp#, yy.getncattr('coordinates'), xx.getncattr('coordinates')


#~ FILLING & CLOSURE OF THE NC (OUPUT) FILE  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def NC_FILE_II( simy ):#, XTRA1, XTRA2 ):
    global timexx, ncvarx#, nctnam
# define the TIME.variable/dimension
    nctnam = f'time_{"{:03d}".format(simy+1)}'        # if more than 100 years/simul are planned
    # nctnam = f'time_{"{:02d}".format(simy+1)}'
    # nctnam = f'time'
    sub_grp.createDimension(nctnam, None)
    # chunkt = CHUNK_3D( [ len(KATE) ], valSize=8)   # because 'i8'
    timexx = sub_grp.createVariable(nctnam, TIMEINT, (nctnam),
        # fill_value=np.array( -1 ).astype( 'i8' ))#, chunksizes=chunkt)
        fill_value=TIMEFIL)#, chunksizes=chunkt)
    # timexx = sub_grp.createVariable(nctnam, 'u8', (nctnam))#, chunksizes=chunkt)
    # timexx[:] = KATE
    timexx.long_name = 'starting time'
    # timexx.units = f'{TIME_OUTNC} since ' + DATE_ORIGEN.strftime('%Y-%m-%d %H:%M:%S')#'%Y-%m-%d %H:%M:%S %Z%z'
    timexx.units = f"{TIME_OUTNC} since {DATE_ORIGEN.strftime('%Y-%m-%d %H:%M:%S')}"
    timexx.calendar = 'proleptic_gregorian'#'gregorian'#
    timexx._CoordinateAxisType = 'Time'
    timexx.coordinates = nctnam

# define the RAINFALL variable
    ncvnam = f'year_{SEED_YEAR+simy}'
    # chunkx = CHUNK_3D( [ len(KATE), len(YS), len(XS) ], valSize=int(RAINFMT[-1]))

    if RAINFMT[0]=='f':
#-DOING.FLOATS------------------------------------------------------------------
        ncvarx = sub_grp.createVariable(ncvnam, datatype=f'{RAINFMT}'
            ,dimensions=(nctnam,'y','x'), zlib=True, complevel=9
            # ,chunksizes=CHUNK_3D( [ len(KATE), len(YS), len(XS) ], valSize=int(RAINFMT[-1]))
            ,least_significant_digit=3, fill_value=np.nan)#least_significant_digit=4
        # ncvarx[:] = KUBE.data.astype(f'{RAINFMT}')
    else:
#-DOING.INTEGERS--------------------------------------------------------------
        ncvarx = sub_grp.createVariable(ncvnam, datatype=f'{RAINFMT}',
            dimensions=(nctnam,'y','x'), zlib=True, complevel=9,
            fill_value=np.array( 0 ).astype( f'{RAINFMT}' ))#, chunksizes=chunkx,least_significant_digit=3)
        # # integer_array = ( (KUBE - ADD) / SCL ).astype(f'{RAINFMT}')
        # # # f'{RAINFMT}' if =='i2'/'i4'/etc converts 'np.nan' (if any) into 0s; so turn those zeros into 'MINIMUM'
        # # integer_array[ np.isnan(integer_array) ] = MINIMUM
        # ncvarx[:] = VOIDXR    # do some silly filling

# THIS IN-MEMORY ROUNDING MIGHT BREAK/SLOW THE SCRIPT!??!
    # ncvarx[:] = (( (KUBE - ADD) / SCL ).round(0)).astype( f'{RAINFMT}' )
    # ncvarx[:] = integer_array
    ncvarx.precision = PRECISION
    # ncvarx.scale_factor = SCL
    # ncvarx.add_offset = ADD
    # ncvarx._FillValue = np.array( MINIMUM ).astype( f'{RAINFMT}' )
    # ncvarx._FillValue = np.array( 0 ).astype( f'{RAINFMT}' )

    ncvarx.units = 'mm'
    ncvarx.long_name = 'rainfall'
    ncvarx.grid_mapping = sref_name

    ncvarx.coordinates = f'{tag_y} {tag_x}'
    #ncvarx.coordinates = f'{yy.getncattr("coordinates")} {xx.getncattr("coordinates")}'

# # define & fill some other XTRA1 variable (previously set up)
# # 'f4' guarantees 1-second (1/60 -minute) precision
#     sub_grp.variables['duration'][:,simy] = ((XTRA1 *60).round(0) /60).astype('f4')
# # # https://stackoverflow.com/a/28425782/5885810  (round to the nearest-nth) -> second
# #     sub_grp.variables['duration'][:,simy] = list(map(lambda x: round(x /(1/60)) *1/60, ass ))
#     sub_grp.variables['sampled_total'][simy] = XTRA2.astype('f4')

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- NC.FILE CREATION ---------------------------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#%% CORE COMPUTATION #1

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
def COMPUTE_LOOP( MASK_SHP, NP_MASK, MRAIN, seas, simy, nreg, MOTHER ):
# MOTHER=allTIME; MASK_SHP=REGIONS['mask'][nreg]; NP_MASK=REGIONS['npma'][nreg]; MRAIN=REGIONS['rain'][nreg]

# # initialize some VOID arrays
# # https://stackoverflow.com/a/22732845/5885810  # 'void' numpys
#     r_ain = np.array([], dtype='u1').reshape(0, len(YS), len(XS))
#     masksum = np.empty( (1), dtype='f8' )
#     masksum = np.empty( (0), dtype='f8' )
#     masksum.fill(np.nan)

    masksum = np.zeros_like( MOTHER )

# for the WGc-case we start with (~40 days/month * 5 months (WET-S1)) = 200
# ...hence, we're assuming that (initially) we have more than 1 storm/day
# ...(then we continue 'half-ing' the above seed)
# 40*4 (FOR '_SF' RUNS); 40*2 (FOR '_SC' RUNS); 40*1 (FOR 'STANDARD' RUNS)
    NUM_S = 40*1 * M_LEN[ seas ].__getitem__(0)
    NUM_S = 555
    CUM_S = 0
    # KOUNT = 0   # TEMPORAL.STOPPING.CRITERION

# DO IT UNTIL THE TOTAL RAINFALL IS REACHED OR NO MORE STORMS TO COMPUTE
    while CUM_S < MRAIN and NUM_S >= 2:
    # while KOUNT < 3 and NUM_S >= 2:     # DOES THE CYCLE 3x MAXIMUM!
#%%
    # sample random storm centres
    # # https://stackoverflow.com/a/69630606/5885810  (rnd pts within shp)
    #     CENTS = poisson( BUFFRX.geometry.xs(0), size=NUM_S )
    #     CENTS = poisson( MASK_SHP.geometry.xs(0), size=NUM_S )
        CENTS = SCENTRES( MASK_SHP, NUM_S )

    # sampling maxima radii
        RADII = TRUNCATED_SAMPLING( RADIUS[ seas ][''], [1* MINRADIUS, None], NUM_S )
        RADII = RADII *9.5 # -> so we can have large storms than the resolution
    # polygon(s) for maximum radii
        RINGO = LAST_RING( RADII, CENTS )#, MINRADIUS )

    # define pandas to split the Z_bands (or not)
        qants, ztats = ZTRATIFICATION( pd.concat( RINGO ) )
    # compute copulas given the Z_bands (or not)
        MAX_I, DUR_S = list(map(np.concatenate, zip(* qants.apply( lambda x:\
            # COPULA_SAMPLING(COPULA, seas, x['E'], x['median']), axis='columns') ) ))
            COPULA_SAMPLING(COPULA, seas, x['E'], x[ Z_STAT ]), axis='columns') ) ))
# REVISE! if the sorting is indeed NECESSARY!!
    # sort back the arrays
        MAX_I, DUR_S = list(map( lambda A: A[ np.argsort( ztats.index ) ], [MAX_I, DUR_S] ))
        DUR_S = DUR_S *1.5

    # SANCTION UNWANTED DURATIONS
        okdur = CHOP( DUR_S )
    # update NUM_S (some DUR_S might have gone)
# https://stackoverflow.com/a/8364723/5885810  -> np.count_nonzero faster than sum()
        NUM_S = np.count_nonzero( okdur )

    # increase/decrease maximum intensites
        # MAX_I = MAX_I[ okdur ] * (1 + STORMINESS_SC[ seas ] + ((simy +0) *STORMINESS_SF[ seas ]))
        MAX_I = MAX_I[ okdur ] * 2.1#1
    # choping any max_intensity above the permitted/designed MAXD_RAIN
        MAX_I[ MAX_I > MAXD_RAIN ] = MAXD_RAIN

# calling the DATE.TIME block
        MATES, i_scaling = QUANTIZE_TIME( NUM_S, seas, simy, DUR_S[ okdur ] )

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


#~ CENTRAL WRAPPER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def STORM( NC_NAMES ):
    global cut_lab, cut_bin, tod_fun, nc, sub_grp, nsim, REGIONS
# define Z_CUTS labelling (if necessary)
    if Z_CUTS:
        cut_lab = [f'Z{x+1}' for x in range(len(Z_CUTS) +1)]
        cut_bin = np.union1d(Z_CUTS, [0, 9999])
# read (and check) the PDF-parameters
    READ_PDF_PAR()
    # CHECK_PDF()
    ALT_CHECK_PDF()
# define some xtra-basics
    WET_SEASON_DAYS()
    SHP_REGION()
# define transformation constants/parameters for INTEGER rainfall-NC-output
    NCBYTES_RAIN()

    # # https://gis.stackexchange.com/a/267326/127894     (get EPSG/CRS from raster)
    # from osgeo import osr
    # tIFF = gdal.Open( DEM_FILE )
    # tIFF = gdal.Open( './data_WG/dem/WGdem_WGS84.tif' )
    # tIFF_proj = osr.SpatialReference( wkt=tIFF.GetProjection() ).GetAttrValue('AUTHORITY', 1)
    # tIFF = None

    print('\nRUN PROGRESS')
    print('************')

# FOR EVERY SEASON
    for seas in range( SEASONS ):# seas=0
    # ESTABLISH HOW THE DOY-SAMPLING WILL BE DONE
        tod_fun = 'TOD_CIRCULAR' if all( list(map(lambda k:
            DOYEAR[ seas ].keys().__contains__(k), ['p', 'mus', 'kappas'])) ) else 'TOD_DISCRETE'
    # CREATE NC.FILE
        ncid = NC_NAMES[ seas ]
        nc = nc4.Dataset(ncid, 'w', format='NETCDF4')
        nc.created_on = datetime.now(tzlocal()).strftime('%Y-%m-%d %H:%M:%S %Z')#%z
        print(f'\nSEASON: {seas+1}/{SEASONS}')

# FOR EVERY SIMULATION
        for nsim in range( NUMSIMS ):# nsim=0
            print(f'\tRUN: {"{:02d}".format(nsim+1)}/{"{:02d}".format(NUMSIMS)}')#, end='', flush=False)
        # 1ST FILL OF THE NC.FILE (defining global vars & CRS)
            sub_grp = NC_FILE_I( nc, nsim )

# FOR EVERY YEAR (of the SIMULATION)
            for simy in tqdm( range( NUMSIMYRS ), ncols=50 ):# simy=0

            # 2ND FILL OF THE NC.FILE (creating the TIME & RAIN vars)
                NC_FILE_II( simy )

                rain_fs = READ_REALIZATION( RAIN_MAP, SUBGROUP, WKT_OGC, YS, XS )
                REGIONS = REGIONALISATION( rain_fs, CLUSTERS, BUFFRX_MASK, CATCHMENT_MASK )
                # REGIONS = REGIONALISATION( rain_fs.rain, CLUSTERS, BUFFRX_MASK, CATCHMENT_MASK, rain_fs.void )

# sampling total seasonal rainfall (a.k.a., STOPPING CRITERIA)...
            # # TOTALP is passed here as eventually Michael wats a TOTALP_PDF per REGION
            #     REGIONS['rain'] = list(map(lambda x:ENHANCE_SR( seas, TOTALP ), range(len( REGIONS['mask'] )) ))
# ... in the meantime just add some made.up values for stopping rain
                backup_rain = REGIONS['rain']
                # # np.random.seed(42)
                # np.random.seed(777)
                # REGIONS.update( {'rain':list(map(np.array, (np.random.rand(CLUSTERS) *100).tolist() ))} )

# # NO NEED for this.updating NO MORE!
#             # add/pass SEAS and SIMY to the REGIONS.dictionary
#                 # https://stackoverflow.com/a/209854/5885810
#                 # https://www.freecodecamp.org/news/add-to-dict-in-python/
#                 REGIONS.update( dict(zip( ['seas','simy'],
#                     list(map(lambda x:[x] *len(REGIONS['rain']), [seas, simy])) )) )

# a void array/variable cannot traverse/GLOBAL
                allTIME = np.array([], dtype='f8')
                cum_OUT = []

                # # https://stackoverflow.com/a/65990898/5885810
                # list(map(COMPUTE_LOOP, *REGIONS.values()))
                for nreg in tqdm( range( CLUSTERS ), ncols=50 ):# nreg=0 nreg=1 nreg=2 nreg=3
                    newmom, cumout = COMPUTE_LOOP( REGIONS['mask'][nreg], REGIONS['npma'][nreg],
                        REGIONS['rain'][nreg], seas, simy, nreg, allTIME )
                    allTIME = newmom
                    # allTIME = NEWMOM
                    cum_OUT.append( cumout )

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
                pd.DataFrame({'k':range(CLUSTERS), 'mean_in':REGIONS['rain'], 'mean_out':cum_OUT
                              ,'mean_xtra':cum_xtra
                    }).to_csv(ncid.replace('.nc','_kmeans.csv'), index=False, mode='a', sep=',')

        nc.close()

#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#-  CORE COMPUTATION & NC.FILE FILLING --------------------------------- (END) #
#-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#%%

if __name__ == '__main__':

    from pathlib import Path
    Path( abspath( join(parent_d, OUT_PATH) ) ).mkdir(parents=True, exist_ok=True)
    # define NC.output file.names
    NC_NAMES =  list(map( lambda a,b,c: f'{abspath( join(parent_d, OUT_PATH) )}/RUN_'\
        f'{datetime.now(tzlocal()).strftime("%y%m%dT%H%M")}_S{a+1}_{b.strip()}_{c.strip()}.nc',\
        # range(SEASONS), PTOT_SCENARIO, STORMINESS_SCENARIO ))
        range(SEASONS), ['nada','zero'], ['zero','nada'] ))

    STORM( NC_NAMES )
    # # testing for only ONE Season!
    # STORM( [f'./model_output/RUN_test-XX.nc'] )