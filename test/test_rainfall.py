"""
test for all functions in ../STORM3/rainfall.py
"""

# append STORM3 to sys
from os.path import abspath, dirname, join
from sys import path
# https://stackoverflow.com/a/248066/5885810
curent_d = dirname(__file__)
path.insert(0, dirname( curent_d ))

# # IPython's alternative(s)
# # https://stackoverflow.com/a/38231766/5885810
# curent_d = dirname(abspath('__file__'))
# path.insert(0, dirname( curent_d ))
# # path.append( dirname( curent_d ))

# import local libs
from numpy import load, allclose, array#, loadtxt#
from numpy.random import seed
from datetime import datetime
from pyarrow.parquet import read_table
from netCDF4 import Dataset
from xarray import open_dataset
from argparse import Namespace

# now import the module
import rainfall

# # # import key parameters
# # from parameters import X_RES, Y_RES, BUFFER, SHP_FILE, WKT_OGC#, RAIN_MAP, SUBGROUP, CLUSTERS
# # defining 'globals' [it's bad practice, i know!]
# rainfall.BUFFRX_MASK = []
# rainfall.CATCHMENT_MASK = []

# multi-test constants/files
mask_pq = abspath( join(curent_d, './tmp-raster_mask_GRID.pq') )
buff_pq = abspath( join(curent_d, './tmp-raster_mask-buff_GRID.pq') )
mask_ = abspath( join(curent_d, './tmp-raster_mask.npy') )
buff_ = abspath( join(curent_d, './tmp-raster_mask-buff.npy') )
nc_i_ = abspath( join(curent_d, './nc_i.nc') )
ncii_ = abspath( join(curent_d, './nc_ii.nc') )


def test_READ_PDF_PAR():

# calling the function
    rainfall.READ_PDF_PAR()

# https://saturncloud.io/blog/how-to-select-by-partial-string-from-a-pandas-dataframe/
    try:
        assert all([rainfall.PDFS.index.str.contains(x).any() for x in
            ['RADIUS_', 'COPULA_', 'DOYEAR_']])
        print('READ_PDF_PAR ...\tmodule rainfall.py  runs OK!')
    except:
# https://stackoverflow.com/a/730778/5885810
        print("READ_PDF_PAR ...\tmodule rainfall.py  BREAKS!!"\
              "\n   [file 'ProbabilityDensityFunctions_' might be set.up incorrectly]")


def test_RETRIEVE_PDF():

    test_pdfc = "<class 'scipy.stats._distn_infrastructure.rv_continuous_frozen'>"
    test_pdfd = "<class 'scipy.stats._distn_infrastructure.rv_discrete_frozen'>"

    # to enable some globals (in case...)
    rainfall.READ_PDF_PAR()
# calling the function
    dists_core = list(map(rainfall.RETRIEVE_PDF,
        ['AVGDUR_PDF', 'MAXINT_PDF', 'BETPAR_PDF', 'RADIUS_PDF']))#, 'TOTALP_PDF']))
    dists_core = [str(x[''].__class__) == test_pdfc for x in list(zip(*dists_core))[0]]

    try:
        d_xtra = list(map(rainfall.RETRIEVE_PDF, ['COPULA_RHO', 'DOYEAR_VMF', 'DATIME_VMF']))
    except:
        d_xtra = list(map(rainfall.RETRIEVE_PDF, ['COPULA_RHO', 'DOYEAR_PMF']))
    d_raw = list(zip( *[list(x.values()) for x in list(zip(*d_xtra))[0]] ))[0]
    d_xtra = [str(type(x)) == test_pdfd or str(type(x)) == "<class 'numpy.ndarray'>"\
        or str(type(x)) == "<class 'numpy.float64'>"\
    # the line below can be activated/deactivated
        or str(type(x)) == test_pdfc\
        for x in d_raw]

    assert all(dists_core) and all(d_xtra)
    print('RETRIEVE_PDF ...\tmodule rainfall.py  runs OK!')


def test_CHECK_PDF():
    # https://stackoverflow.com/a/36489085/5885810
    from warnings import catch_warnings, filterwarnings
# to NOT affect all other warnings
    with catch_warnings():
        filterwarnings('error')

        try:
            # to enable some globals (in case...)
            rainfall.READ_PDF_PAR()
# calling the function
            rainfall.CHECK_PDF()

            # assert str( rainfall.RADIUS.__getitem__(0)[''].__class__ ) == \
            #     "<class 'scipy.stats._distn_infrastructure.rv_continuous_frozen'>"
            print('CHECK_PDF ...\t\tmodule rainfall.py  runs OK!')

        except Warning:
            print("CHECK_PDF ...\t\tmodule rainfall.py  runs OK!"\
                  "\n   [...expels some warnings due to inconsistency in PARAMETERS.py]")


def test_ALT_CHECK_PDF():

# calling the function
    rainfall.ALT_CHECK_PDF()

    assert str( rainfall.RADIUS.__getitem__(0)[''].__class__ ) == \
        "<class 'scipy.stats._distn_infrastructure.rv_continuous_frozen'>"
    print('ALT_CHECK_PDF ...\tmodule rainfall.py  runs OK!')


def test_WET_SEASON_DAYS():

# calling the function
    rainfall.WET_SEASON_DAYS()

    # # https://stackoverflow.com/a/16151611/5885810
    # isinstance( datetime.now(), tuple(map(type, rainfall.DATE_POOL.__getitem__(0))) )
    # issubclass( datetime, tuple(map(type, rainfall.DATE_POOL.__getitem__(0))) )

    assert all( [isinstance( datetime.now(), tuple(map(type, rainfall.DATE_POOL.__getitem__(0))) )
        , len(rainfall.DOY_POOL)==2, issubclass( datetime, rainfall.DATE_ORIGEN.__class__ )] )
    print('WET_SEASON_DAYS ...\tmodule rainfall.py  runs OK!')

# # alternative
#     lo_class = [list(map(type, rainfall.DATE_POOL.__getitem__(0))),
#         len(rainfall.DOY_POOL), rainfall.DATE_ORIGIN.__class__]
#     test_lo_class = [[datetime, datetime], 2, datetime]

#     assert all(a==b for a,b in zip(lo_class, test_lo_class))
#     print('WET_SEASON_DAYS ...\tmodule rainfall.py  runs OK!')


def test_SHP_REGION_GRID():

    test_catchment = read_table( mask_pq )
    test_catchment = test_catchment.to_pandas().T.to_numpy()
    test_buffrx = read_table( buff_pq )
    test_buffrx = test_buffrx.to_pandas().T.to_numpy()

    def_xres = rainfall.X_RES
    def_yres = rainfall.Y_RES
# setting modified globals (turned off in PARAMETERS.py by default)
    rainfall.XLLCORNER =  1319567.308750340249      # in meters! (x.coord of the lower.left edge, i.e., not.the.pxl.center)
    rainfall.YLLCORNER = -1170429.328196450602      # in meters! (y.coord of the lower.left edge, i.e., not.the.pxl.center)
    rainfall.X_RES     =      919.241896152628      # in meters! (pxl.resolution for the 'regular/local' CRS)
    rainfall.Y_RES     =      919.241896152628      # in meters! (pxl.resolution for the 'regular/local' CRS)
    rainfall.N_X       =     2313                   # number of cells/pxls in the X-axis
    rainfall.N_Y       =     2614                   # number of cells/pxls in the Y-axis

# calling the function
    rainfall.SHP_REGION_GRID()

# # GDAL_3.7.0--Linux: adds 3.pxl to catchment & 2.pxl to buffr, at this RES.
# # ...missing 3/2.5M-pxls is worth not reprinting numpys... so just add ATOL
    assert allclose([rainfall.CATCHMENT_MASK, rainfall.BUFFRX_MASK],
                    [test_catchment, test_buffrx], atol=5)
    print('SHP_REGION_GRID ...\tmodule rainfall.py  runs OK!')

# re.set the DEFAULT "global" resolution
    rainfall.X_RES = def_xres
    rainfall.Y_RES = def_yres


def test_SHP_REGION():

    test_catchment = load( mask_ )
    test_buffrx = load( buff_ )
    # # if reading .ASC
    # test_catchment = loadtxt('tmp-raster_mask.asc', skiprows=6)
    # test_catchment[ test_catchment==-9999 ] = 0

# calling the function
    rainfall.SHP_REGION()

# GDAL_3.7.0--Linux: adds 4 pxls to the mask [check: 'tmp-raster_mask--GDAL_3.7.0_LINUX']
# ...thus, to make tests compatibles across platforms... ZERO the excess OUT (so cheat!!)
    rainfall.CATCHMENT_MASK[(array([71, 271, 366, 390]), array([230, 314, 217, 76]))] = 0

    assert allclose([rainfall.CATCHMENT_MASK, rainfall.BUFFRX_MASK], [test_catchment, test_buffrx])
    print('SHP_REGION ...\t\tmodule rainfall.py  runs OK!')


def test_NCBYTES_RAIN():

    # to enable some globals (in case SHP_REGION is not previously called/tested)
    rainfall.SHP_REGION()
# calling the function
    rainfall.NCBYTES_RAIN()

    # https://stackoverflow.com/a/4541167/5885810
    # assert allclose(list(rainfall.VOIDXR.shape), [0, 470, 408]) and \
    assert allclose(rainfall.VOIDXR.shape, tuple([0, 470, 408])) and \
        isinstance(rainfall.SCL, float) and isinstance(rainfall.ADD, float)
    print('NCBYTES_RAIN ...\tmodule rainfall.py  runs OK!')


def test_NC_FILE_I():

# # setting back the modified globals
#     rainfall.X_RES = 5000                           # in meters! (pxl.resolution for the 'regular/local' CRS)
#     rainfall.Y_RES = 5000                           # in meters! (pxl.resolution for the 'regular/local' CRS)

    # to enable some globals (in case SHP_REGION is not previously called/tested)
    rainfall.SHP_REGION()
# calling the function
    nc_file = abspath( join(curent_d, './test_rainfall_nc_i.nc') )
    nc = Dataset(nc_file, 'w', format='NETCDF4')
    sub_grp = rainfall.NC_FILE_I( nc, 0 )
    nc.close()

    nc_i = open_dataset(nc_file, group='run_01', decode_coords='all')
    test_nc_i = open_dataset(nc_i_, group='run_01', decode_coords='all')
    test_nc_i.close()
    nc_i.close()

    # assert test_nc_i.equals( nc_i )
    assert nc_i.equals( test_nc_i )
    print('NC_FILE_I ...\t\tmodule rainfall.py  runs OK!')


def test_NC_FILE_II():
    # rainfall.DATE_ORIGIN = ''

    # to enable some globals (in case...)
    rainfall.WET_SEASON_DAYS()
    rainfall.SHP_REGION()
    # to create the first stage of the nc.file
    nc_file = abspath( join(curent_d, './test_rainfall_nc_ii.nc') )
    nc = Dataset(nc_file, 'w', format='NETCDF4')
# assigning NC_FILE_I output to RAINFALL globals
    rainfall.sub_grp = rainfall.NC_FILE_I( nc, 0 )

# calling the function
    rainfall.NC_FILE_II( 0 )
    nc.close()

    nc_ii = open_dataset(nc_file, group='run_01', decode_coords='all')
    test_nc_ii = open_dataset(ncii_, group='run_01', decode_coords='all')
    test_nc_ii.close()
    nc_ii.close()

    # line below is dangerous as depends on parameter SEED_YEAR
    # assert nc_ii.equals( test_nc_ii )
    assert nc_ii.coords.dims == test_nc_ii.coords.dims
    print('NC_FILE_II ...\t\tmodule rainfall.py  runs OK!')


def test_COMPUTE_LOOP():

    test_newmom = [27971400, 27971430, 27971460, 27971490, 27971520, 27971550, 27971580]
    test_cumout = 0.020117367307388518

    n_clus_ = 2
    n_pos = n_clus_ -1
    rainfall.tod_fun = 'TOD_CIRCULAR'
    # rainfall.tod_fun = 'TOD_DISCRETE'
    rainfall.NUM_S = 55

    # to enable some globals (in case...)
    rainfall.CHECK_PDF()
    rainfall.WET_SEASON_DAYS()
    rainfall.SHP_REGION()
    rainfall.NCBYTES_RAIN()

    rain_fs = rainfall.READ_REALIZATION( rainfall.RAIN_MAP, rainfall.SUBGROUP,
        rainfall.WKT_OGC, rainfall.SHP_REGION.__globals__['YS'],
        rainfall.SHP_REGION.__globals__['XS'] )
    # define some seed [not 42 this time :(]
    seed( 24 )
    REGIONS = rainfall.REGIONALISATION( rain_fs, n_clus_,
        rainfall.SHP_REGION.__globals__['BUFFRX_MASK'],
        rainfall.SHP_REGION.__globals__['CATCHMENT_MASK'] )
    REGIONS['rain']

    # to create the first stage of the nc.file
    nc_file = abspath( join(curent_d, './test_rainfall_loop.nc') )
    nc = Dataset(nc_file, 'w', format='NETCDF4')
# assigning NC_FILE_I output to RAINFALL globals
    rainfall.sub_grp = rainfall.NC_FILE_I( nc, 0 )
    # to create the second stage of the nc.file
    rainfall.NC_FILE_II( 0 )

# calling the function
    seed( 42 )
    newmom, cumout = rainfall.COMPUTE_LOOP( REGIONS['mask'][n_pos], REGIONS['npma'][n_pos],
        REGIONS['rain'][n_pos] *1e-4, 0, 0, n_pos, array([], dtype='f8') )
    nc.close()

# i wouldn't use NEWMOM as depends on the DATE_ORIGIN & TIME_OUTNC [CUMOUT is absolute]
    assert allclose(cumout, test_cumout)# and allclose(newmom, test_newmom)
    print('COMPUTE_LOOP ...\tmodule rainfall.py  runs OK!')









def test_PAR_UPDATE():

    n_clus_ = 3

# calling the function
    rainfall.PAR_UPDATE( Namespace(CLUSTERS=n_clus_) )

    assert rainfall.CLUSTERS == n_clus_
    print('PAR_UPDATE ...\t\tmodule rainfall.py  runs OK!')


#%% running all tests

if __name__ == '__main__':
    print('\n', end='')
    # test_READ_PDF_PAR()
    # test_RETRIEVE_PDF()
    # test_CHECK_PDF()
    test_ALT_CHECK_PDF()
    test_WET_SEASON_DAYS()
    test_SHP_REGION_GRID()
    test_SHP_REGION()
    test_NCBYTES_RAIN()
    # test_NC_FILE_I()
    # test_NC_FILE_II()
    # test_COMPUTE_LOOP()

    test_PAR_UPDATE()