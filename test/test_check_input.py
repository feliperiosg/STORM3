"""
test for all functions in ../STORM3/check_input.py
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
from argparse import Namespace
from pandas import DataFrame

# now import the module
import check_input


# unitestS start here!
# --------------------
def test_PAR_UPDATE():

    n_clus_ = 3

# calling the function
    # check_input.PAR_UPDATE( Namespace(SEASONS=1, NUMSIMS=2, NUMSIMYRS=1,
    #     PTOT_SC=[0.0], PTOT_SF=[0.0], STORMINESS_SC=[-0.0], STORMINESS_SF=[0.0]) )
    check_input.PAR_UPDATE( Namespace(CLUSTERS=n_clus_) )

    # assert check_input.PAR_UPDATE.__globals__['CLUSTERS'] == n_clus_
    assert check_input.CLUSTERS == n_clus_
    print('PAR_UPDATE ...\t\tmodule check_input.py  runs OK!')


def test_ASSERT():

# calling the function
    try:
        check_input.ASSERT()
        print('ASSERT ...\t\tmodule check_input.py  runs OK!')
    except:
# https://stackoverflow.com/a/730778/5885810
        print("ASSERT ...\t\tmodule check_input.py  BREAKS!!"\
              "\n   [but it's because the PARAMETERS.py file is set.up incorrectly]")


def test_INFER_SCENARIO():

    test_str_vec = [['ptotS-', 'ptotT+'], ['stormsC'], ['n/a']]

    tab_sign  = DataFrame({'Var1':['', '+', '-']                            ,'Var2':[0, 1,-1   ]})
    tab_ptot  = DataFrame({'Var1':['ptotC', 'ptotS', 'ptotT', 'n/a']        ,'Var2':[0, 1, 2, 3]})
    # tab_ptot  = DataFrame({'Var1':['ptotC', 'ptotS', 'ptotT', 'special']    ,'Var2':[0, 1, 2, 3]})
    tab_storm = DataFrame({'Var1':['stormsC', 'stormsS', 'stormsT', 'n/a']  ,'Var2':[0, 1, 2, 3]})

# calling the function
    str_vec = [ check_input.INFER_SCENARIO( [ -1/3, 0], [0, +3/5], tab_ptot, tab_sign )
        , check_input.INFER_SCENARIO( [None, 2/3], [None, -9/10], tab_storm, tab_sign )
        , check_input.INFER_SCENARIO( [2/3, None], [-9/10, None], tab_storm, tab_sign ) ]

    assert test_str_vec == str_vec
    print('INFER_SCENARIO ...\tmodule check_input.py  runs OK!')


def test_WELCOME():

# calling the function
    try:
        check_input.ASSERT()

# https://stackoverflow.com/a/8391735/5885810   (blocking PRINT)
        import sys
        from os import devnull

        sys.stdout = open(devnull, 'w')
# calling the function
        names = check_input.WELCOME()
        sys.stdout = sys.__stdout__

        assert check_input.OUT_PATH in names.__getitem__(0) and names.__getitem__(0)[-3:] == '.nc'
        print('WELCOME ...\t\tmodule check_input.py  runs OK!')

    except:
# https://stackoverflow.com/a/730778/5885810
        print("WELCOME ...\t\tmodule check_input.py  BREAKS!!"\
              "\n   [due to the PARAMETERS.py file set.up incorrectly; i told you!]"\
              "\n   [... go back, modify PARAMETERS.py, and test will run smoothly]")


#%% running all tests

if __name__ == '__main__':
    print('\n', end='')
    test_PAR_UPDATE()
    test_ASSERT()
    test_INFER_SCENARIO()
    test_WELCOME()
