#%% SOFT-CORE PARAMETERS

"""
Parameters in this block define the type, (temporal) extend, and a couple of
climatic/meterological conditions of the desired RUN.
The parameters set up here will be the default input of STORM.
You can either modify/tweak them here (thus avoiding passing them again when
running STORM from the command line) or passing/defining them righ from the
command line whe running STORM.
For an 'in-prompt' help (on these parameters) type:
    "python storm.py -h"    (from your CONDA environment or Terminal)
    "%%python storm.py -h"  (from your Python console)
"""

# SEASONS = 1#1             # Number of Seasons (per Run)
NUMSIMS = 1#2         # Number of runs per Season
NUMSIMYRS = 1#2         # Number of years per run (per Season)
"""
*** perhaps, either NUMSIMS or NUMSIMYRS or both has/have to be re-thought?,
    because i can't really have multiple SIMULATED-YEARS each one containing
    multiple SIMULATION-NUMBERS/exercises, each one of (potential) 1000's of
    half-hourly time-steps/stamps... it'll take a lot of space in the NC.output!
"""

"""
PTOT_SC       = Signed scalar specifying the step change in the observed wetness (TOTALP)
PTOT_SF       = Signed scalar specifying the progressive trend in the observed wetness
STORMINESS_SC = Signed scalar specifying the step change in the observed storminess
STORMINESS_SF = Signed scalar specifying the progressive trend in the observed storminess
*** all scalars must be specified between 0 and 1 (i.e., 100%) ***
*** a (un-)signed scalar implies stationary conditions akin to (current) observations ***
"""

# # PARAMETER = [ S1 ]
PTOT_SC       = [0.00]
PTOT_SF       = [ 0.0]
STORMINESS_SC = [-0.0]
STORMINESS_SF = [+0.0]
# # PARAMETER = [ S1 ]
PTOT_SC       = 0.00
PTOT_SF       = -.06
STORMINESS_SC =  0.05
STORMINESS_SF = +0.0


#%% HARD-CORE PARAMETERS

"""
Parameters in this block define the input and output files (paths), and the
spatio-temporal characteristics of the domain over which STORM will run.
Unlike the parameters set up in the previous block, these parameters cannot
be passed from the command line. Therefore, their modification/tweaking must
carried out here.
"""

PDF_FILE = './model_input/ProbabilityDensityFunctions.csv'  # pdf.pars file
# SHP_FILE = './model_input/KC_regions.shp' # regions.by.Katherine.Cocking (WGS84)
SHP_FILE = './model_input/HAD_basin.shp'  # catchment shape-file in WGS84
DEM_FILE = './model_input/HAD_wgs84.tif'  # aoi raster-file (optional**)
# DEM_FILE = None
ZON_FILE = './model_input/regions.shp'  # nK regions export file
OUT_PATH = './model_output'                             # output folder
"""
**  DEM_FILE is only required for runs at different altitudes, i.e., Z_CUTS != None
*** Having the DEM_FILE in the local CRS could contribute to a faster run,
    although we didn't find staggering differences in both approaches.
    Still, if the preferred option is a local-CRS DEM, switch ON/OFF the line(s):
    'zonal_stats(vectors=Z_OUT.geometry, raster=DEM_FILE, stats=Z_STAT) in 'ZTRATIFICATION'.
"""

# RAIN_MAP = '../3B-HHR.MS.MRG.3IMERG.20101010-S100000-E102959.0600.V06B.HDF5'   # no.CRS at all!
# RAIN_MAP = './realisation_MAM_crs-wrong.nc'  # no..interpretable CRS
RAIN_MAP = './model_input/rainfall_MAM.nc'  # yes.interpretable CRS
RAIN_MAP = './model_input/rainfall_OND.nc'  # yes.interpretable CRS
NREGIONS = 4  # number of regions to split the whole.region into
"""
NREGIONS ==1 means no splitting at all!.
The model still 'splits' the region into 1 big area equal to the catchment.
"""

Z_STAT = 'mean'#'median'    # statistic to retrieve from the DEM ['median'|'mean' or 'min'|'max'?? not 'count']
# Z_CUTS = [ 400, 1000]  # [34.2, 67.5]%
# Z_CUTS = [1000, 2000, 3000]  # [67.53, 97.15, 99.87]%
Z_CUTS = [300,  600, 1200]  # in meters! [28.13, 48.75, 78.57]%
# Z_CUTS = None  # (or Z_CUTS = []) for INT-DUR copula modelling regardless altitude
"""
Z_CUTS = [1350, 1500] are from the analysis carried out in the 'pre_processing.py' module.
They imply 3 altitude-bands, namely, [0, 1350), [1350, 1500), [1500, 9999), for which the
Intensity-Duration copulas were established.
Hence, modyfing this variable without a copula (re-)analysis, for the desired/updated
bands, will still yield results!; nevertheless, such results won't be representative
of the parametric functions/statistics found in '.../ProbabilityDensityFunctions.csv'.
"""

TACTIC = 1  # the "way" STORM must be run, i.e., [1] for NO copulas; [2] COPULAS

# # EPSG:42106 is not implemented in PyProj (or anywhere else) xP
# EPSG_CODE = 42106       # EPSG Code of the local/regular Coordinate Reference System (https://epsg.io/42106)
# PROJ4_STR = '+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
# PROJ4_STR = '+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +ellps=sphere +units=m +no_defs +type=crs'

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

### GRID characterization
BUFFER    =  7000.                      # in meters! -> buffer distance (out of the HAD)
X_RES     =  5000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
Y_RES     =  5000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
# X_RES     = 10000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
# Y_RES     = 10000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
# X_RES     =  1000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
# Y_RES     =  1000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
"""
BUFFER extends the catchment boundary (some given distance), thus delimiting the area
for which the storm centres are generated (within).
The extension (bounding-box) of this 'buffer' defines too the limits of the rainfall
fields, namely, the Area Of Interest (aoi).
"""

### TEMPORAL characterization
T_RES   =  30                           # in minutes! -> TEMPORAL.RES of TIME.SLICES
NO_RAIN =  0.01                         # in mm -> minimum preceptible/measurable/meaningful RAIN in all AOI
MIN_DUR =  20                            # in minutes!
MAX_DUR =  60 * 24 * 4                  # in minutes! -> 4 days (in this case)
# # OR:
# MIN_DUR =  []                           # use 'void' arrays if you want NO.CONSTRAINT on storm-duration
# MAX_DUR =  []                           # ... in either (or both) MIN_/MAX_DUR parameters/constants
"""
MAX_DUR and MIN_DUR constraints the storm-duration of the sampled pairs
from the intenstity-duration copula.
"""
# designed MAXIMUM rainfall for T_RES!! (Twice of that of IMERG!) [note also that 120 << 131.068_iMAX]
MAXD_RAIN = 60 * 2                      # in mm
DISPERSE_ = .2                          # factor to split MAXD_RAIN into

SEASON_TAG = 'OND'
# # these parameters allow to pin down a time-dimension to the storms
# SEED_YEAR  = None                       # for your SIM/VAL to start in the current year
SEED_YEAR = 2024                        # for your SIM/VAL to start in 2050
TIME_ZONE      = 'Africa/Addis_Ababa'               # Local Time Zone (see links below for more names)
# # OR:
# TIME_ZONE    = 'UTC'
# # https://stackoverflow.com/a/64861179/5885810    (zoneinfo list)
# # https://pynative.com/list-all-timezones-in-python/#h-get-list-of-all-timezones-name
# # https://www.timeanddate.com/time/map/
# # https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
# DATE_ORIGIN    = '1950-01-01'                       # to store dates as INT
DATE_ORIGIN    = '1970-01-01'                       # to store dates as INT
"""
SEASONS_MONTHS = [[6,10], None] (equal to Z_CUTS) corresponds to the period (and season!)
for which all the parametric functions found in '.../ProbabilityDensityFunctions.csv' were
computed. Hence, when you modify this variable (for whatever your needs), please carry out
all the respective (stochastic) analyses correspond for the period you want to model.
"""

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
TIMEFIL = +(2**(int(TIMEINT[-1]) * 8)) - 1
TIME_OUTNC = 'minutes'  # UNITS (since DATE_ORIGIN) for NC.TIME dim
# TIME_DICT_ = dict(seconds=60 ,minutes=1, hours=1/60, days=(60*24)**-1)
TIME_DICT_ = dict(seconds=1, minutes=1/60, hours=1/60**2, days=1/(60**2*24))
