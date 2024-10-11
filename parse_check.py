# https://stackoverflow.com/a/248066/5885810
from os.path import abspath, dirname, join, exists
parent_d = dirname(__file__)    # otherwise, will append the path.of.the.tests
# parent_d = './'               # to be used in IPython

import numpy as np
from pandas import DataFrame
from pathlib import Path
from datetime import datetime
from dateutil.tz import tzlocal
from parameters import *

# https://stackoverflow.com/a/23116937/5885810  (0 divison -> no warnings)
# https://stackoverflow.com/a/29950752/5885810  (0 divison -> no warnings)
np.seterr(divide='ignore', invalid='ignore')


# %% update parameters

#~ replace FILE.PARAMETERS with those read from the command line ~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# def PAR_UPDATE(args):
#     # https://stackoverflow.com/a/2083375/5885810  (exec global... weird)
#     for x in list(vars(args).keys()):
#         exec(f'globals()["{x}"] = args.{x}')
#     # print([PTOT_SC, PTOT_SF])

def ARG_UPDATE( args ):
    for x in list(vars( args ).keys()):
        # # use this if you're not inside a (local) function
        # exec(f'if args.{x}:\n\t{x} = args.{x}')
# https://stackoverflow.com/a/2083375/5885810   (exec global... weird)
        exec(f'if args.{x}:\n\tglobals()["{x}"] = args.{x}')
    return args

# so one can input NONE from the command line
# https://stackoverflow.com/a/48295546/5885810  (None in argparse)
def none_too( v ):
    return None if v.lower() == 'none' else float( v )


# %% parsing

#~ read parameters from the command line ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# https://www.geeksforgeeks.org/command-line-arguments-in-python/
# https://docs.python.org/3/library/argparse.html#the-add-argument-method
def PARCE( parser ):
# add the defaults
# https://stackoverflow.com/a/27616814/5885810  (parser case insensitive)
    parser.add_argument('-w', '--SEASONS', type=int, default=SEASONS, choices=[1,2],
        help='Number of Seasons (per Run) (default: %(default)s)')
    parser.add_argument('-n', '--NUMSIMS', type=int, default=NUMSIMS,
        help='Number of runs per Season (default: %(default)s)')
    parser.add_argument('-y', '--NUMSIMYRS', type=int, default=NUMSIMYRS,
        help='Number of years per run (per Season) (default: %(default)s)')
    parser.add_argument('-ps', '--PTOT_SC', default=PTOT_SC, type=none_too, nargs='+',#type=float
        help='Relative change in the seasonal rain equally applied to every '\
            'simulated year. (one signed scalar per Season).')
    parser.add_argument('-pf', '--PTOT_SF', default=PTOT_SF, type=none_too, nargs='+',
        help='Relative change in the seasonal rain progressively applied to '\
            'every simulated year. (one signed scalar per Season).')
    parser.add_argument('-ss', '--STORMINESS_SC', default=STORMINESS_SC, type=none_too,
        nargs='+', help='Relative change in the observed intensity equally applied '\
            'to every simulated year. (one signed scalar per Season).')
    parser.add_argument('-sf', '--STORMINESS_SF', default=STORMINESS_SF, type=none_too,
        nargs='+', help='Relative change in the observed intensity progressively '\
            'applied to every simulated year. (one signed scalar per Season).')
    parser.add_argument('--version', action='version', version='STORM 3.')#'%(prog)s 2.0')
# Read arguments from command line
    args = parser.parse_args()
    # print(args)
# REDEFINE variables by their names, instead of relative to 'args'
    updated_args = ARG_UPDATE( args )

    return updated_args


# %% some

# wet_hash = DataFrame({'Var2':['PTOT', 'STORMINESS'],
#                       'Var3':['Total Rainfall', 'Rain Intensity']})


#~ BLURTS OUT WARNINGS IF YOUR 'SOFT-CORE' PARAMETERS 'SMELL FUNNY' ~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def ASSERT(wet_hash):
# checks the SEASONS (input) parameters
    assert SEASONS in (1, 2), 'SEASONS not valid!\nIt must be either 1 or 2.'

# checks the _STEPCHANGE & _SCALING_FACTOR (input) parameters
    for j in wet_hash.Var2:#j='PTOT'
        SC = f'{j}_SC'
        SF = f'{j}_SF'
    # do STEPCHANGE & SCALING_FACTOR have the same lengths??
        assert len(eval(SC)) == len(eval(SF)), f'Imcompatible Sizes!\nPlease, '\
            f'ensure that both {SC} and {SF} have the same length/size.'
    # does each dimension/season have the same length among them??
        assert np.in1d([np.unique( tuple([np.asarray(x).size for x in eval(x)]) ).size\
            for x in [SC, SF]] ,1).all(),\
                'Imcompatible Sizes!\nPlease, ensure that the parameteres (either'\
                f' in {SC} or {SF}) have the same length/size for each season.'
    # is each dimension/season as long as 1 or NUMSIMYRS??
        for i in range(SEASONS):#i=0
            assert 1 in tuple(map( lambda x: np.asarray(eval(f'{x}[{i}]')).size, [SC, SF] )) or\
                NUMSIMYRS in tuple(map( lambda x: np.asarray(eval(f'{x}[{i}]')).size, [SC, SF] )),\
                    f'Imcompatible Sizes!\nBoth {SC} and {SF} must have '\
                    'lengths of either 1 (one parameter for all NUMSIMYRS)'\
                    f' or {NUMSIMYRS} (one parameter for each NUMSIMYRS).'
# https://stackoverflow.com/a/25050572/5885810  (1st items of a list of lists)
# https://stackoverflow.com/a/31154423/5885810  (find None in tuples)
# https://stackoverflow.com/q/12116491/5885810  (0's to None)
        # the list comprehension has to be done as Python 'reads' 0s as None
            assert not any( map( lambda x: x is None,\
                list( zip(*map( eval, [SC, SF] )) ).__getitem__(i) ) ), 'Missing Values!'\
                    '\nThere are missing values (i.e. None) in some (or all) of the'\
                    f' {", ".join([SC, SF])} variables for Season {i+1}.\nPlease,'\
                    ' ensure that the aforementioned variables contain numerical '\
                    f'values, if you are indeed planning to model Season {i+1}.'
        # does the progression factor (SF) add to 1.0 or more than 1.0??
            assert abs( eval(f'{SF}[{i}]') *NUMSIMYRS ) <1,\
                f'Scalar Overflooding!\nPlease, ensure that over the {NUMSIMYRS}-year,'\
                f' the cumulative sum of the progressive trend scalar {SF} be always less than 1.0.'
        # is the scalar factor (SC) larger (or equal) than 1.0??
            assert abs( eval(f'{SC}[{i}]') ) <1, f'Scalar Overflooding!\nPlease, '\
                f'ensure a value less than 1.0 for any step change scalar in {SC}.'
        some_sum = np.nansum((
            np.asarray(eval(SC), dtype='f8') / np.asarray(eval(SC), dtype='f8'),
            2* np.asarray(eval(SF), dtype='f8') / np.asarray(eval(SF), dtype='f8')
            ), axis=0, dtype=np.int32)
    # the integer 3 helps to distinguish between '...stepchange' & '...scaling_factor'
        assert 3 not in some_sum, f'{j.upper()}_SCENARIO not valid!\n'\
            f'Please, ensure that either {SC} or {SF} (or both!) are '\
            f'set to 0 (zero), for any given season of the run.'

# check that the DEM exists (if you're using Z_CUTS)
    if Z_CUTS:
        assertdem = f'NO DEM_FILE!\n'\
            'You chose to model rainfall at/for different altitudes, i.e., Z_CUTS != [].'\
            f' Nevertheless, the path to the DEM ({DEM_FILE}) was not correctly set up '\
            'or the file does not exist.\nPlease ensure that the DEM file exists in the'\
            ' correct path.\nConversely, you can also switch off the Z_CUTS variable'\
            ' (i.e., Z_CUTS == [] or None) if you aim to model rainfall regardless altitude.'
# https://stackoverflow.com/a/82852/5885810     (files exists?)
# https://stackoverflow.com/a/40183030/5885810  (assertion error)
# https://stackoverflow.com/a/6095782/5885810   (multiple excepts)
# https://stackoverflow.com/a/46827349/5885810  (None-Type error)
        # try:
        #     Path( abspath( join(parent_d, DEM_FILE) ) ).resolve( strict=True )
        # except (FileNotFoundError, TypeError):
        #     raise AssertionError( assertdem )
        # #---
        # try:
        #     join(parent_d, DEM_FILE)
        # except (AttributeError, TypeError):
        #     raise AssertionError( assertdem )
# https://stackoverflow.com/a/60324874/5885810  (using os.path.exists)
# https://therenegadecoder.com/code/how-to-check-if-a-file-exists-in-python/
        if DEM_FILE is None or not exists( abspath( join(parent_d, DEM_FILE) ) ):
            raise AssertionError( assertdem )

# check that the SHP exists (always!)
    assertshp = f'NO SHP_FILE!\n'\
        f'The path to the SHP file ({SHP_FILE}) was not correctly set up or the file'\
        ' does not exist.\nPlease ensure that the SHP file exists in the correct path.'
    # try:
    #     Path( abspath( join(parent_d, SHP_FILE) ) ).resolve( strict=True )
    # except (FileNotFoundError, TypeError):
    #     raise AssertionError( assertshp )
    if SHP_FILE is None or not exists( abspath( join(parent_d, SHP_FILE) ) ):
        raise AssertionError( assertshp )

# check that the SEASON_MONTHS exists and/or are correctly set up
    assertext = f'You chose to model {SEASONS} season(s) but there is/are either'\
        ' missing or not correctly allocated seasonal period(s) in the variable '\
        'SEASONS_MONTHS, which defines the date-times of the wet season(s).\n'\
        'Please update the aforementioned variable accordingly.'
    if SEASONS == 1:
        assert list(map(lambda x: not None in x, zip(SEASONS_MONTHS))).__getitem__(0), assertext
    else:
        assert not None in SEASONS_MONTHS, assertext


class welcome:

    def __init__(self, **kwargs):
        """
        generates and prints the names of the output nc-files.\n
        Input: none.\n
        **kwargs ->
        ptot_sc : list; Step Change factors in observed wetness.
        ptot_sf : list; Progressive Trend factors in observed wetness.
        storm_sc : list; Step Change factors in observed storminess.
        storm_sf : list; Progressive Trend factors in observed storminess.
        n_sims : int;
        n_sims_y : int;
        out_path : char;\n
        Output -> list; containing output file-paths/names.
        """

        # assign the global/default parameters (to inner variables)
        self.ptot_sc = kwargs.get('ptot_sc', PTOT_SC)
        self.ptot_sf = kwargs.get('ptot_sf', PTOT_SF)
        self.storm_sc = kwargs.get('storm_sc', STORMINESS_SC)
        self.storm_sf = kwargs.get('storm_sf', STORMINESS_SF)
        self.numsims = kwargs.get('n_sims', NUMSIMS)
        self.numsimyrs = kwargs.get('n_sims_y', NUMSIMYRS)
        self.out_path = kwargs.get('out_path', OUT_PATH)
        # table relating core variables (Var2) to understandable meanings (Var3)
        self.wet_hash = DataFrame({
            # 'Var2':['PTOT', 'STORMINESS'],
            'Var2':['ptot', 'storm'],
            'Var3':['Total Rainfall', 'Rain Intensity']
            })
        # signs (Var1) related to scalars (Var2)
        self.tab_sign = DataFrame({
            'Var1': ['', '+', '-'],
            'Var2': [0, 1, -1],
            })
        # tables to correlate signs & scenarios (for both scaling factors)
        self.tab_ptot = DataFrame({
            'Var1': ['ptotC', 'ptotS', 'ptotT', 'n/a'],
            'Var2': [0, 1, 2, 3],
            })
        self.tab_storm = DataFrame({
            'Var1': ['stormsC', 'stormsS', 'stormsT', 'n/a'],
            'Var2': [0, 1, 2, 3],
            })
        """
      'ptotC' = Stationary conditions / Control Climate
      'ptotS' = Step Change (increase/decrese) in observed wetness
      'ptotT' = Progressive Trend (positive/negative) in observed wetness
    'stormsC' = Stationary conditions / Control Climate
    'stormsS' = Step Change (increase/decrese) in observed storminess'
    'stormsT' = Progressive Trend (positive/negative) in observed storminess'
        'n/a' = scenario NOT DEFINED (as both xxx_SC & xxx_SF differ from 0)
        """
        self.ncs = self.welcome()

    def welcome(self,):
        # global ptot_scene, storm_scene
        """
        generates the names of the output nc-files.\n
        Input: none.\n
        Output -> list; containing output file-paths/names.
        """
        # infer scenarios
        ptot_scene = self.infer_scenario(
            self.ptot_sc, self.ptot_sf, self.tab_ptot, self.tab_sign
            )
        storm_scene = self.infer_scenario(
            self.storm_sc, self.storm_sf, self.tab_storm, self.tab_sign
            )

        # create OUT_PATH folder (if it doen'st exist already)
        # https://stackoverflow.com/a/50110841/5885810  (create folder if exisn't)
        abs_path = abspath(join(parent_d, self.out_path))
        Path(abs_path).mkdir(parents=True, exist_ok=True)
        # define NC.output file.names
        nc_paths =  list(map(
            lambda a, b, c: f'{Path(abs_path)}/'
            f'{datetime.now(tzlocal()).strftime("%y%m%dT%H%M")}_sim{"{:02d}".format(a+1)}_{b.strip()}_'
            f'{c.strip()}.nc', range(self.numsims), ptot_scene, storm_scene))

        # print the CORE INFO
        print('\nRUN SETTINGS')
        print('************\n')
        print(f'number of simulations: {self.numsims}')
        print(f'years per simulation : {self.numsimyrs}')
        for j in self.wet_hash.Var2:
            # var = 'SC' if eval(f'{j}_SCENARIO')[-2]=="S" else 'SF'  # NOT ENTIRELY ACCURATE!!
            print(f'{self.wet_hash[self.wet_hash.Var2.isin([j])].Var3.iloc[0]} scenarios '
                  f'({" | ".join([f"S{x+1}" for x in range(self.numsims)])}):  '
            # 8 because 'stormsT+' is the maximum length of these strings
                  f'{ " | ".join(map(eval, [f"{j}_scene[{x}].center(8," ")" for x in range(self.numsims)]))}')
        # https://stackoverflow.com/a/25559140/5885810  (string no sign)
        # https://www.delftstack.com/howto/python/python-pad-string-with-spaces/
        # https://stackoverflow.com/a/45120812/5885810  (print/create string padding)
        print('\nOutput paths:')
        print(*[(k.ljust(max(map(len, nc_paths)), ' ')).rjust(
            max(map(len, nc_paths)) + 4, ' ') for k in nc_paths], sep='\n')
        return nc_paths

    def infer_scenario(self, stepchange, scaling_factor, tab_x, tab_sign):
    # stepchange=PTOT_SC; scaling_factor=PTOT_SF; tab_x=tab_ptot
        """
        transforms the numerical input of sc/sf factors into 'readable' labels.\n
        Input ->
        *stepchange* : list; list of floats with step_change factors/coeffs.
        *scaling_factor* : list; list of floats with scaling factors/coeffs.
        *tab_x* : pandas.DataFrame; scenario-codes (Var1) related to scalars (Var2).\n
        Output -> list; input-size list with the scenario-tags to append to file-name.
        """
        # convert input into numpy
        stepchange = np.asarray(stepchange, dtype='f8')
        scaling_factor = np.asarray(scaling_factor, dtype='f8')
        # establish whether is a 0, 1, 2, or 3
        sum_vec = np.nansum(
            (stepchange / stepchange, 2 * scaling_factor / scaling_factor),
            axis=0, dtype=np.int32)
        # compute signs
        sign_ar = np.sign(np.sign(stepchange) + np.sign(scaling_factor))
        # find the variables and their signs int the corresponding 'tables'
        # the 'for' loop is necessary as neither .isin nor .reindex can be used...
        # (negative or repeated values)
        # https://stackoverflow.com/a/51327154/5885810  (.reindex instead of .isin)
        str_vec = list(map(
            lambda A, B: f"{A}{B}", tab_x.loc[sum_vec, 'Var1'].values,
            np.concatenate([tab_sign.loc[tab_sign.Var2.isin([x]), 'Var1'].values
                            for x in sign_ar[~np.isnan(sign_ar)]])
            ))
        return str_vec


# %% run

# if __name__ == '__main__':
#     willkommen = welcome()
#     ASSERT(willkommen.wet_hash)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'STOchastic Rainstorm Model [STORM v3.0]')
    PARCE(parser)
    willkommen = welcome()
    ASSERT(willkommen.wet_hash)
