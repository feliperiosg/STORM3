"""
test for all functions in ../STORM3/realization.py
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
from numpy import load, allclose
from numpy.ma import MaskedArray
from numpy.random import seed
from xarray import open_dataarray

# now import the module (and some help)
import realization
from rainfall import SHP_REGION
# run SHP_REGION to define some GlobalS
SHP_REGION()

# multi-test constants/files
real_file = load( abspath( join(curent_d, './realisation.npy') ) )
void_file = abspath( join(curent_d, './void.nc') )


def test_EMPTY_MAP():

    test_void = open_dataarray(void_file, decode_coords='all')

# calling the function
    void = realization.EMPTY_MAP( SHP_REGION.__globals__['YS'],
        SHP_REGION.__globals__['XS'], SHP_REGION.__globals__['WKT_OGC'] )

    # assert test_void.equals( void )
    assert void.equals( test_void )
    print('EMPTY_MAP ...\t\tmodule realization.py  runs OK!')


def test_READ_REALIZATION():

# calling the function
    real = realization.READ_REALIZATION( SHP_REGION.__globals__['RAIN_MAP'],
        SHP_REGION.__globals__['SUBGROUP'], SHP_REGION.__globals__['WKT_OGC'],
        SHP_REGION.__globals__['YS'], SHP_REGION.__globals__['XS'] )

    assert allclose(real.rain.data, real_file)
    print('READ_REALIZATION ...\tmodule realization.py  runs OK!')


def test_KREGIONS():

    test_sum_cdic = [238.03293]

# calling the function
    seed( 42 )
    mask_regn = MaskedArray( real_file.copy(), ~SHP_REGION.__globals__['CATCHMENT_MASK'].astype('bool') )
    npreg, cdic = realization.KREGIONS( mask_regn, N_C=4 )

    pxls = [len(npreg[npreg==x]) for x in range(len(cdic))]
    sum_cdic = sum([a*b for a,b in zip(list(cdic.values()), [x/sum(pxls) for x in pxls])])

    assert allclose(sum_cdic, test_sum_cdic)
    print('KREGIONS ...\t\tmodule realization.py  runs OK!')


def test_MORPHOPEN():

    test_pxls = [34337, 17975, 29499, 4309]

    from numpy import isnan
    # computing prerequisites
    seed( 42 )
    mask_regn = MaskedArray( real_file.copy(), ~SHP_REGION.__globals__['CATCHMENT_MASK'].astype('bool') )
    npreg, cdic = realization.KREGIONS( mask_regn, N_C=4 )
    npreg[ isnan(npreg) ] = -9999

# calling the function
    npreg = realization.MORPHOPEN( npreg )

    pxls = [len(npreg[npreg==x]) for x in range(len(cdic))]

    assert allclose(pxls, test_pxls)
    print('MORPHOPEN ...\t\tmodule realization.py  runs OK!')


def test_REGIONALISATION():

    n_clus_ = 2
    test_lo = [['MultiPolygon'] *n_clus_, [176.6739, 426.87433], 21069]

# calling the function
    real = realization.READ_REALIZATION( SHP_REGION.__globals__['RAIN_MAP'],
        SHP_REGION.__globals__['SUBGROUP'], SHP_REGION.__globals__['WKT_OGC'],
        SHP_REGION.__globals__['YS'], SHP_REGION.__globals__['XS'] )
    # define some seed [42 obviously]
    seed( 42 )
    output = realization.REGIONALISATION( real, n_clus_,
        SHP_REGION.__globals__['BUFFRX_MASK'], SHP_REGION.__globals__['CATCHMENT_MASK'] )

    lo = [output['rain'], output['kmeans'][ output['kmeans']!=-1 ].sum()]

    assert all( list(map(allclose, lo, test_lo[1:])) ) and \
        list(map(lambda x:x.geom_type.__getitem__(0), output['mask'])) == test_lo[0]
    print('REGIONALISATION ...\tmodule realization.py  runs OK!')

    # # https://www.geeksforgeeks.org/python-check-if-two-lists-are-identical/
    # assert lo==test_lo
    # assert all(a==b for a,b in zip(lo, test_lo[1:]))


#%% running all tests

if __name__ == '__main__':
    print('\n', end='')
    test_EMPTY_MAP()
    test_READ_REALIZATION()
    test_KREGIONS()
    # test_MORPHOPEN()
    test_REGIONALISATION()