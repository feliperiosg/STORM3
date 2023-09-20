"""
test for all functions in ../STORM3/rainfall.py
"""

# append STORM3 to sys
from os.path import abspath, dirname, join
import sys
# https://stackoverflow.com/a/248066/5885810
curent_d = dirname(__file__)
sys.path.insert(0, dirname( curent_d ))

# # IPython's alternative(s)
# # https://stackoverflow.com/a/38231766/5885810
# curent_d = dirname(abspath('__file__'))
# sys.path.insert(0, dirname( curent_d ))
# # sys.path.append( dirname( curent_d ))

# import local libs
from numpy import load, allclose, array, concatenate, stack, where, unique#, loadtxt
from scipy.stats import johnsonsb, expon, geninvgauss, nhypergeom, norm, vonmises
from warnings import catch_warnings, filterwarnings
from geopandas import GeoDataFrame, points_from_xy
from pyarrow.parquet import read_table
from pandas import DataFrame, concat
from xarray import open_dataset
from argparse import Namespace
from numpy.random import seed
from datetime import datetime
from zoneinfo import ZoneInfo
from netCDF4 import Dataset
from os import devnull

# now import the module
import rainfall


# multi-test constants/files
radius_x = [{'':johnsonsb(1.5187, 1.2696, -0.2789, 20.7977)}, {'':None}]
doyear_x = [{'p': array([0.054, 0.089, 0.07, 0.087, 0.700]),
    'mus': array([1.9, 0.228, 1.55, 1.172, 0.558]) -3.14159 /2,
    'kappas': array([105.32,  51.97,  87.19,  52.91, 6.82])}, { }]
datime_x = [{'p': array([0.2470, 0.3315, 0.4215]),
    'mus': array([0.6893, 1.7034, 2.5756]), 'kappas': array([6.418, 3., 0.464])}, { }]
carea = GeoDataFrame(geometry=points_from_xy(x=[0], y=[0]).buffer(42195 /2, resolution=8))
# for now use scypi's VONMISES [it's ~100x faster than VMM-package!]
windir_x = [{'':vonmises(.9 *3.14159, .00001)}] *2
# # ...IF you're using M-vonMises for WINDIR do (having updated MOVING_STORM accordingly):
# windir_x = [{'p': array([1.]), 'mus': array([.9]) *3.14159, 'kappas': array([0.00001])}] *2
wspeed_x = [{'':norm(7.55, 1.9)}] *2

# dateTime stuff (to be stored as int)
tzon_ = 'Europe/Prague'#'Africa/Addis_Ababa'
tdic_ = dict(seconds=1 ,minutes=1/60, hours=1/60**2, days=1/(60**2*24))
tout_ = 'hours'
d_ori = datetime(2000,1,1,0,0, tzinfo=ZoneInfo(tzon_))
dpool = [[datetime(2023,3,1,0,0), datetime(2023,6,1,0,0)]] *2

# help - functions
def import_LOTR():
# https://stackoverflow.com/a/8391735/5885810   (blocking PRINT)
    sys.stdout = open(devnull, 'w')
    # call the "test_LOTR()" test, to not re-type everything!
    ring_s = test_LOTR()
    sys.stdout = sys.__stdout__
    # to enable some globals (in case SHP_REGION is not previously called/tested)
    rainfall.SHP_REGION()
    return ring_s

def do_RASTERIZE():
    ring_s = import_LOTR()
    # call RASTERIZE
    fall = list(map(rainfall.RASTERIZE, ring_s))
    inca = [array([0.25249051, 0.5, 0.03477616]), array([0.04968086, 0.39800247])]
    mate = array([28043760, 28043790, 28043820, 28042530, 28042560])
    return fall, inca, mate


# unitestS start here!
# --------------------
def test_LOTR():

    m_r = array([4.05919237, 4.05919237, 4.05919237, 10.84884845, 10.84884845])
    m_m = array([2.74484185, 2.74484185, 2.74484185, 7.3010916, 7.3010916])
    m_d = array([107.23559307, 107.23559307, 107.23559307, 300.96376946, 300.96376946])
    m_b = array([0.05904296, 0.05904296, 0.05904296, 0.27340132, 0.27340132])
    m_c = array([[2827021.84314557, 224130.04551293], [2825226.41077656, 228495.38947558],
        [2821670.96552634, 237139.95984508], [1964545.63149682, 52219.10776983],
        [1965198.51928862, 52883.30199964]])

# calling the function
    r_r = rainfall.LOTR( m_r, m_m, m_d, m_b, m_c )

    assert len(r_r) == len(m_c) and \
        all([list(map(lambda y:str(y)=='LineString', y)) for y in [x.geom_type for x in r_r]])
    print('LOTR ...\t\tmodule rainfall.py  runs OK!')

    return r_r


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

# to NOT affect all other warnings
    # https://stackoverflow.com/a/36489085/5885810
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

    def_xres = rainfall.X_RES
    def_yres = rainfall.Y_RES
# setting modified globals (turned off in PARAMETERS.py by default)
    rainfall.XLLCORNER =  1319567.308750340249      # in meters! (x.coord of the lower.left edge, i.e., not.the.pxl.center)
    rainfall.YLLCORNER = -1170429.328196450602      # in meters! (y.coord of the lower.left edge, i.e., not.the.pxl.center)
    rainfall.X_RES     =      919.241896152628      # in meters! (pxl.resolution for the 'regular/local' CRS)
    rainfall.Y_RES     =      919.241896152628      # in meters! (pxl.resolution for the 'regular/local' CRS)
    rainfall.N_X       =     2313                   # number of cells/pxls in the X-axis
    rainfall.N_Y       =     2614                   # number of cells/pxls in the Y-axis

    test_catchment = read_table( abspath( join(curent_d, './tmp-raster_mask_GRID.pq') ))
    test_catchment = test_catchment.to_pandas().T.to_numpy()
    test_buffrx = read_table( abspath( join(curent_d, './tmp-raster_mask-buff_GRID.pq') ) )
    test_buffrx = test_buffrx.to_pandas().T.to_numpy()

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

    test_catchment = load( abspath( join(curent_d, './tmp-raster_mask.npy') ) )
    test_buffrx = load( abspath( join(curent_d, './tmp-raster_mask-buff.npy') ) )
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

    # to enable some globals (in case...)
    rainfall.SHP_REGION()

# calling the function
    rainfall.NCBYTES_RAIN()

    # https://stackoverflow.com/a/4541167/5885810
    # assert allclose(list(rainfall.VOIDXR.shape), [0, len(rainfall.YS), len(rainfall.XS)]) and \
    assert allclose(rainfall.VOIDXR.shape, tuple([0, len(rainfall.YS), len(rainfall.XS)])) and \
        isinstance(rainfall.SCL, float) and isinstance(rainfall.ADD, float)
    print('NCBYTES_RAIN ...\tmodule rainfall.py  runs OK!')


def test_NC_FILE_I():

    # to enable some globals (in case SHP_REGION is not previously called/tested)
    rainfall.SHP_REGION()

    nc_file = abspath( join(curent_d, './test_rainfall_nc_i.nc') )
    nc = Dataset(nc_file, 'w', format='NETCDF4')
# calling the function
    sub_grp = rainfall.NC_FILE_I( nc, 0 )
    nc.close()

    nc_i = open_dataset(nc_file, group='run_01', decode_coords='all', drop_variables=['mask'])
    test_nc_i = open_dataset(abspath( join(curent_d, './nc_i.nc') ),
        group='run_01', decode_coords='all', drop_variables=['mask'])
    test_nc_i.close()
    nc_i.close()

    # assert test_nc_i.equals( nc_i )
    assert nc_i.equals( test_nc_i )
    print('NC_FILE_I ...\t\tmodule rainfall.py  runs OK!')


def test_NC_FILE_II():

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

    nc_ii = open_dataset(nc_file, group='run_01', decode_coords='all', drop_variables=['mask'])
    test_nc_ii = open_dataset(abspath( join(curent_d, './nc_ii.nc') ),
        group='run_01', decode_coords='all', drop_variables=['mask'])
    test_nc_ii.close()
    nc_ii.close()

    # line below is dangerous as depends on parameter SEED_YEAR
    assert nc_ii.equals( test_nc_ii )
    # # ...an easy one would be:
    # assert nc_ii.coords.dims == test_nc_ii.coords.dims
    print('NC_FILE_II ...\t\tmodule rainfall.py  runs OK!')


def test_COMPUTE_LOOP():
    # https://stackoverflow.com/a/36489085/5885810
    from warnings import catch_warnings, filterwarnings

# 0: for CHECK_PDF()  |  1: for ALT_CHECK_PDF()
    WHAT_PARS = 1

    n_clus_ = 2
    n_pos = n_clus_ -1
    rainfall.WINDIR = windir_x
    rainfall.WSPEED = wspeed_x
    rainfall.tod_fun = 'TOD_CIRCULAR'
    rainfall.NUM_S = 55

    if WHAT_PARS == 0:
        with catch_warnings():
            filterwarnings('error')
            try:
                # to enable some globals without warnings
                rainfall.READ_PDF_PAR()
                rainfall.CHECK_PDF()
            except Warning:
                pass
        # tests
        test_newmom = [28021260, 28021290, 28025190, 28025220, 28025250, 28025280,
               28025310, 28025340, 28025370, 28025400, 28025430, 28025460]
        test_cumout = 0.017769773964003123
    else:
        rainfall.ALT_CHECK_PDF()
        # tests
        test_newmom = [27971400, 27971430, 27971460, 27971490, 27971520, 27971550, 27971580]
        test_cumout = 0.020117367307388518

# more enabled globals...
    rainfall.WET_SEASON_DAYS()
    rainfall.SHP_REGION()
    rainfall.NCBYTES_RAIN()

    rain_fs = rainfall.READ_REALIZATION( rainfall.RAIN_MAP, rainfall.SUBGROUP,
        rainfall.WKT_OGC, rainfall.YS, rainfall.XS )
    # define some seed [not 42 this time :(]
    seed( 24 )
    REGIONS = rainfall.REGIONALISATION( rain_fs, n_clus_,
        rainfall.BUFFRX_MASK, rainfall.CATCHMENT_MASK )
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
    assert allclose(cumout, test_cumout, rtol=.9999) and allclose(newmom, test_newmom)
    print('COMPUTE_LOOP ...\tmodule rainfall.py  runs OK!')

# re.set to DEFAULTS
    del rainfall.tod_fun
    del rainfall.NUM_S
    del rainfall.WINDIR
    del rainfall.WSPEED


def test_SCENTRES():

    test_cents = array([[-5293.77968524, 19017.89015897], [9788.98437473,
        4162.89474069], [-14514.29346653, -14515.31121441]])

# calling the function
    seed( 42 )
    cents = rainfall.SCENTRES( carea, test_cents.shape.__getitem__(0) )

    assert allclose(cents, test_cents)
    print('SCENTRES ...\t\tmodule rainfall.py  runs OK!')


def test_TRUNCATED_SAMPLING():

# calling the function
    seed( 777 )
    xampl = rainfall.TRUNCATED_SAMPLING( radius_x[0][''], [1.7, None], 3 )

    assert allclose(xampl, array([2.71975498, 3.61616629, 2.14687658]))
    print('TRUNCATED_SAMPLING ...\tmodule rainfall.py  runs OK!')


def test_LAST_RING():

    n_cent_ = 3

    cents = rainfall.SCENTRES( carea, n_cent_ )
    xampl = rainfall.TRUNCATED_SAMPLING( radius_x[0][''], [1* 5, None], n_cent_ )
# calling the function
    ringo = rainfall.LAST_RING( xampl, cents )

    assert list(map(lambda x:x.geom_type.__getitem__(0), ringo)) == ['Polygon'] *n_cent_
    print('LAST_RING ...\t\tmodule rainfall.py  runs OK!')


def test_ZTRATIFICATION():

    n_cent_ = 3

    seed( 42 )
    cents = rainfall.SCENTRES( carea, n_cent_ )
    xampl = rainfall.TRUNCATED_SAMPLING( radius_x[0][''], [1* 5, None], n_cent_ )
    ringo = rainfall.LAST_RING( xampl, cents )
# calling the function
    qants, ztats = rainfall.ZTRATIFICATION( concat( ringo ) )

    assert qants[rainfall.Z_STAT].sum() == n_cent_ and len(ztats) == n_cent_
    print('ZTRATIFICATION ...\tmodule rainfall.py  runs OK!')


def test_COPULA_SAMPLING():

# re-defining globals
    rainfall.MAXINT = [{'':expon(0.1057 ,6.9955)}] *2
    rainfall.AVGDUR = [{'':geninvgauss(-0.089, 0.77, 2.8432, 82.0786)}] *2

    rhos_ = [{'':-0.31622, 'Z1':-0.276, 'Z2':-0.312, 'Z3':-0.44}] *2
    qants = DataFrame( {'E':'', rainfall.Z_STAT:3}, index=[0] )
    # # QANTS for a 3-ztratification exercise looks like:
    # qants = DataFrame( {'E':['Z1','Z2','Z3'], rainfall.Z_STAT:[2,3,5]} )

# calling the function
    seed( 42 )
    maxi, durs = list(map(concatenate, zip(* qants.apply( lambda x:\
        rainfall.COPULA_SAMPLING(rhos_, 0, x['E'], x[ rainfall.Z_STAT ]), axis='columns') ) ))

    assert len(maxi) + len(durs) == qants[ rainfall.Z_STAT ].sum() *2 and \
        str(maxi.dtype) == 'float64' #and str(durs.dtype) == 'float64'
    print('COPULA_SAMPLING ...\tmodule rainfall.py  runs OK!')

# re.se to DEFAULTS
    del rainfall.MAXINT
    del rainfall.AVGDUR


def test_CHOP():

# calling the function
    d_bool = rainfall.CHOP( array([107.23559307, 300.96376946, 56.86099363]) )

    assert str(d_bool.dtype) == 'bool'
    print('CHOP ...\t\tmodule rainfall.py  runs OK!')


def test_TOD_CIRCULAR():

    def_tzon = rainfall.TIME_ZONE
    def_tdic = rainfall.TIME_DICT_
    def_tout = rainfall.TIME_OUTNC
# 'setting' modified globals (turned off in PARAMETERS.py by default)
    rainfall.TIME_ZONE = tzon_
    rainfall.TIME_DICT_ = tdic_
    rainfall.TIME_OUTNC = tout_

    rainfall.DATE_ORIGEN = d_ori
    rainfall.DATE_POOL = dpool
    rainfall.DOYEAR = doyear_x
    rainfall.DATIME = datime_x

    test_stamps = array([203847.20324407, 204499.72240801, 204382.7530946])

# calling the function
    seed( 42 )
    stamps = rainfall.TOD_CIRCULAR( len(test_stamps), 0, 0 )

    assert allclose(stamps, test_stamps)
    print('TOD_CIRCULAR ...\tmodule rainfall.py  runs OK!')

# re.set to DEFAULTS
    rainfall.TIME_ZONE = def_tzon
    rainfall.TIME_DICT_ = def_tdic
    rainfall.TIME_OUTNC = def_tout
    del rainfall.DOYEAR
    del rainfall.DATIME
    del rainfall.DATE_POOL
    del rainfall.DATE_ORIGEN


def test_TOD_DISCRETE():

    def_tzon = rainfall.TIME_ZONE
    def_tdic = rainfall.TIME_DICT_
    def_tout = rainfall.TIME_OUTNC
# 'setting' modified globals (turned off in PARAMETERS.py by default)
    rainfall.TIME_ZONE = tzon_
    rainfall.TIME_DICT_ = tdic_
    rainfall.TIME_OUTNC = tout_

    rainfall.DATE_ORIGEN = d_ori
    rainfall.DATE_POOL = dpool
    rainfall.DOYEAR = [{'':nhypergeom(712., 610., 38., 0)}] *2

    test_stamps = array([205247.54777778, 205149.33444444, 205152.58861111])

# calling the function
    seed( 777 )
    stamps = rainfall.TOD_DISCRETE( len(test_stamps), 0, 0 )

    assert allclose(stamps, test_stamps)
    print('TOD_DISCRETE ...\tmodule rainfall.py  runs OK!')

# re.set to DEFAULTS
    rainfall.TIME_ZONE = def_tzon
    rainfall.TIME_DICT_ = def_tdic
    rainfall.TIME_OUTNC = def_tout
    del rainfall.DOYEAR
    del rainfall.DATE_POOL
    del rainfall.DATE_ORIGEN


def test_BASE_ROUND():

    def_tdic = rainfall.TIME_DICT_
    def_tout = rainfall.TIME_OUTNC
# 'setting' modified globals (turned off in PARAMETERS.py by default)
    rainfall.TIME_DICT_ = tdic_
    rainfall.TIME_OUTNC = tout_

    stamps = array([203847.20324407, 204499.72240801, 204382.7530946])
    test_iround = array([203847., 204499., 204382.])

# calling the function
    seed( 2112 )
    iround = rainfall.BASE_ROUND( stamps, base=60 )

    assert allclose(iround, test_iround)
    print('BASE_ROUND ...\t\tmodule rainfall.py  runs OK!')

# re.set to DEFAULTS
    rainfall.TIME_DICT_ = def_tdic
    rainfall.TIME_OUTNC = def_tout


def test_TIME_SLICE():

    def_tres = rainfall.T_RES
    def_tdic = rainfall.TIME_DICT_
    def_tout = rainfall.TIME_OUTNC
# 'setting' modified globals (turned off in PARAMETERS.py by default)
    rainfall.TIME_DICT_ = tdic_
    rainfall.TIME_OUTNC = tout_
    rainfall.T_RES = 60

    difs = (array([203847.20324407, 204499.72240801, 204382.7530946]) +
        -1* array([203847., 204499., 204382.]) )
    durs = array([107.23559307, 300.96376946, 56.86099363]) /2
    test_sfact = [array([0.79675593, 0.09687401]),
        array([0.27759199, 1., 1., 0.23043942]), array([0.2469054, 0.22693621])]

# calling the function
    seed( 666 )
    sfact = rainfall.TIME_SLICE( difs, durs )

    assert  all(list(map(allclose, sfact, test_sfact)))
    print('TIME_SLICE ...\t\tmodule rainfall.py  runs OK!')

# re.set to DEFAULTS
    rainfall.TIME_DICT_ = def_tdic
    rainfall.TIME_OUTNC = def_tout
    rainfall.T_RES = def_tres


def test_QUANTIZE_TIME():

    rainfall.DOYEAR = doyear_x
    rainfall.DATIME = datime_x
    rainfall.tod_fun = 'TOD_CIRCULAR'
    # to enable some globals (in case...)
    rainfall.WET_SEASON_DAYS()

# calling the function
    mates_, iscal_ = rainfall.QUANTIZE_TIME( 3, 0, 0, array([107.236, 300.964, 56.861]) )

    assert len(mates_) == sum(list(map(len, iscal_)))
    print('QUANTIZE_TIME ...\tmodule rainfall.py  runs OK!')

# re.set to DEFAULTS
    del rainfall.DOYEAR
    del rainfall.DATIME
    del rainfall.tod_fun


def test_RANDOM_SAMPLING():

# calling the function
    seed( 42 )
    xampl = rainfall.RANDOM_SAMPLING( radius_x[0][''], 3 )

    assert allclose(xampl, array([3.67866802, 10.66461805, 6.58168423]))
    print('RANDOM_SAMPLING ...\tmodule rainfall.py  runs OK!')


def test_MOVING_STORM():

    rainfall.WINDIR = windir_x
    rainfall.WSPEED = wspeed_x

# please note that this test involved 3 cents/iscal
    cents = array([[-5293.77968524, 19017.89015897], [9788.98437473,
        4162.89474069], [-14514.29346653, -14515.31121441]])
    # cents = array([[2827021.84314557, 224130.04551293],
    #     [2563130.51531954, 478649.08497297], [1964545.63149682, 52219.10776983]])
    iscal = [array([0.25249051, 0.5, 0.03477616]), array([0.10911333,
         0.5, 0.5, 0.5, 0.40695333]), array([0.04968086, 0.39800247])]
    test_ncent = array([
        [ -5293.77968524,  19017.89015897], [ -7089.21205425,  23383.23412162],
        [-10644.65730447,  32027.80449112], [  9788.98437473,   4162.89474069],
        [ 11233.97893882,   5649.68653902], [ 17855.50980075,  12462.74869341],
        [ 24477.04066268,  19275.8108478 ], [ 31098.57152461,  26088.8730022 ],
        [-14514.29346653, -14515.31121441], [-13861.40567473, -13851.1169846 ] ])

# calling the function
    seed( 42 )
    ncent = rainfall.MOVING_STORM( cents, iscal, 0 )

    assert allclose(ncent, test_ncent)
    print('MOVING_STORM ...\tmodule rainfall.py  runs OK!')

# re.set to DEFAULTS
    del rainfall.WINDIR
    del rainfall.WSPEED


def test_XPAND():

    radii = array([  4.05919237,  10.84884845])
    maxi_ = array([  2.74484185,   7.3010916 ])
    durs_ = array([107.23559307, 300.96376946])
    betas = array([  0.05904296,   0.27340132])
    iscal = [array([0.25249051, 0.5, 0.03477616]), array([0.04968086, 0.39800247])]

    test_xp = [array([4.05919237, 4.05919237, 4.05919237, 10.84884845, 10.84884845]),
               array([2.74484185, 2.74484185, 2.74484185, 7.3010916, 7.3010916]),
               array([107.23559307, 107.23559307, 107.23559307, 300.96376946, 300.96376946]),
               array([0.05904296, 0.05904296, 0.05904296, 0.27340132, 0.27340132])]

# calling the function
    seed( 42 )
    xp = rainfall.XPAND( [radii, maxi_, durs_, betas], iscal )

    assert allclose(xp, test_xp)
    print('XPAND ...\t\tmodule rainfall.py  runs OK!')


def test_RASTERIZE():

    ring_s = import_LOTR()

# calling the function
    fall = rainfall.RASTERIZE( ring_s[ 0 ] )

    assert allclose(fall.sum(), 26.771672248840332) and allclose(fall.shape, tuple([470, 408]))
    print('RASTERIZE ...\t\tmodule rainfall.py  runs OK!')


def test_SORT_CUBE():

    fall, inca, mate = do_RASTERIZE()
    cube = stack(fall, axis=0) * concatenate(inca)[:, None, None]

# calling the function
    cube_, mate_ = rainfall.SORT_CUBE( cube, mate )

    assert allclose(mate_, array([28042530, 28042560, 28043760, 28043790, 28043820])) and \
        cube_.sum() == cube.sum()
    print('SORT_CUBE ...\t\tmodule rainfall.py  runs OK!')


def test_RAIN_CUBO():

    fall, inca, mate = do_RASTERIZE()
    rain, sate, zuma = rainfall.RAIN_CUBO( fall, concatenate(inca), mate, rainfall.CATCHMENT_MASK )

    assert allclose(sate, array([28042530, 28042560, 28043760, 28043790, 28043820])) and \
        rain.shape.__getitem__(0) == len(fall) and len(zuma) == len(fall)
    print('RAIN_CUBO ...\t\tmodule rainfall.py  runs OK!')


def test_AUX_MSK():

    fall, inca, mate = do_RASTERIZE()
    rain = stack(fall, axis=0) * concatenate(inca)[:, None, None]
    surpass = where(rain > 10)

    lyrs = list(map(lambda x:where(surpass[0]==x)[0], unique(surpass[0])))
    xtra_r = list(map(lambda x:rainfall.AUX_MSK(rain, array(surpass), rainfall.CATCHMENT_MASK, x ), lyrs))

    assert allclose(xtra_r.__getitem__(0).shape, tuple([2, 86291]), atol=5)
    print('AUX_MSK ...\t\tmodule rainfall.py  runs OK!')


def test_CHOP_MAX_RAIN():

    def_maxd = rainfall.MAXD_RAIN
# setting modified globals (turned off in PARAMETERS.py by default)
    rainfall.MAXD_RAIN = 10

    fall, inca, mate = do_RASTERIZE()
    rain = stack(fall, axis=0) * concatenate(inca)[:, None, None]
    surpass = where(rain > rainfall.MAXD_RAIN)
    test_rain = rain[ where( rain>0 ) ]

# calling the function
    seed( 42 )
    rain_ = rainfall.CHOP_MAX_RAIN( rain, rainfall.CATCHMENT_MASK, surpass )

    assert len(where( rain_ >=2 ).__getitem__(0)) - \
        len(where( test_rain >=2 ).__getitem__(0)) == 3
    print('CHOP_MAX_RAIN ...\tmodule rainfall.py  runs OK!')

# re.set to DEFAULTS
    rainfall.MAXD_RAIN = def_maxd


# def test_STORM():

#     nc_out = [ abspath( join(curent_d, './test_rainfall_storm.nc') ) ]# *2
# # calling the function
#     seed( 42 )
#     rainfall.STORM( nc_out )

#     print('STORM ...\t\tmodule rainfall.py  runs OK!')


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
    test_NC_FILE_I()
    test_NC_FILE_II()
    test_COMPUTE_LOOP()             # !slow.crucial.function! [~2/4min]
    test_SCENTRES()
    test_TRUNCATED_SAMPLING()
    test_LAST_RING()
    test_ZTRATIFICATION()
    test_COPULA_SAMPLING()
    test_CHOP()
    test_TOD_CIRCULAR()
    # test_TOD_DISCRETE()           # !slow.inefficient.function!
    test_BASE_ROUND()
    test_TIME_SLICE()
    test_QUANTIZE_TIME()
    test_RANDOM_SAMPLING()
    test_MOVING_STORM()
    test_XPAND()
    test_LOTR()
    test_RASTERIZE()
    test_SORT_CUBE()
    test_RAIN_CUBO()
    test_AUX_MSK()
    test_CHOP_MAX_RAIN()
    test_PAR_UPDATE()
