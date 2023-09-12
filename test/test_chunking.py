"""
test for all functions in ../STORM3/chunking.py
"""

# append STORM3 to sys
from os.path import abspath, dirname
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
from numpy import allclose

# now import the module
import chunking


def test_BINLIST():

    assert allclose(chunking.BINLIST( 2, 3 ), [0, 1, 0])
    print('BINLIST ...\t\tmodule chunking.py  runs OK!')


def test_PERTURB_SHAPE():

# calling the function
    ccand = chunking.PERTURB_SHAPE( [256.0, 1.0, 1.0], 2 )

    assert allclose(ccand, [256.0, 2.0, 1.0])
    print('PERTURB_SHAPE ...\tmodule chunking.py  runs OK!')


def test_CHUNK_3D():

# calling the function
    vorshape = chunking.CHUNK_3D( [666, 470, 408], valSize=2)

    assert allclose(vorshape, [2, 30, 26])
    print('CHUNK_3D ...\t\tmodule chunking.py  runs OK!')


#%% running all tests

if __name__ == '__main__':
    print('\n', end='')
    test_BINLIST()
    test_PERTURB_SHAPE()
    test_CHUNK_3D()
    # print('\n', end='')