"""
test for all functions in ../STORM3/parse_input.py
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
from argparse import Namespace, ArgumentParser

# now import the module
import parse_input


def test_ARG_UPDATE():

    n_clus_ = 3

# calling the function
    # parse_input.ARG_UPDATE( Namespace(SEASONS=1, NUMSIMS=2, NUMSIMYRS=1,
    #     PTOT_SC=[0.0], PTOT_SF=[0.0], STORMINESS_SC=[-0.0], STORMINESS_SF=[0.0]) )
    parse_input.ARG_UPDATE( Namespace(CLUSTERS=n_clus_) )

    # assert check_input.PAR_UPDATE.__globals__['CLUSTERS'] == n_clus_
    assert parse_input.CLUSTERS == n_clus_
    print('ARG_UPDATE ...\tmodule parse_input.py  runs OK!')


def test_none_too():

    assert list(map(parse_input.none_too, ['NoNE', '42'])) == [None, 42.0]
    print('none_too ...\tmodule parse_input.py  runs OK!')


def test_PARCE():

# calling the function
    updated_args = parse_input.PARCE( ArgumentParser(description = 'STORM3') )

    # test_updated_args = Namespace(SEASONS=1, NUMSIMS=2, NUMSIMYRS=1,
    #     PTOT_SC=[0.0], PTOT_SF=[0.0], STORMINESS_SC=[-0.0], STORMINESS_SF=[0.0])
    # assert updated_args == test_updated_args

    assert isinstance(updated_args.SEASONS, int) and \
        isinstance(updated_args.PTOT_SC.__getitem__(0), float)
    print('PARCE ...\tmodule parse_input.py  runs OK!')


#%% running all tests

if __name__ == '__main__':
    print('\n', end='')
    test_ARG_UPDATE()
    test_none_too()
    test_PARCE()