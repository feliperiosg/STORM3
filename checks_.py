import numpy as np
from pandas import DataFrame
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from warnings import warn
# from parameters import *
from parameters import NUMSIMS, NUMSIMYRS, PTOT_SC, PTOT_SF
from parameters import STORMINESS_SC, STORMINESS_SF
from parameters import OUT_PATH, DEM_FILE, SHP_FILE, Z_CUTS
from dateutil.tz import tzlocal
# https://stackoverflow.com/a/248066/5885810
from os.path import abspath, dirname, join, exists
parent_d = dirname(__file__)  # otherwise, will append the path.of.the.tests

# https://stackoverflow.com/a/23116937/5885810  (0 divison -> no warnings)
# https://stackoverflow.com/a/29950752/5885810  (0 divison -> no warnings)
np.seterr(divide='ignore', invalid='ignore')


# %% parsing

class parse:

    def __init__(self, parser, **kwargs):
        """
        preliminars before running STORM3.\n
        Parameters
        ----------
        parser : argparse.ArgumentParser;
            parameters passed/captured from the command line.\n
        **kwargs
        --------
        n_sims : int; indicates the number of simulations/files to output.
        n_sims_y : int; indicates the number of years (per simulation) to run.
        ptot_sc : list; Step Change factors in observed wetness.
        ptot_sf : list; Progressive Trend factors in observed wetness.
        storm_sc : list; Step Change factors in observed storminess.
        storm_sf : list; Progressive Trend factors in observed storminess.\n
        Returns
        -------
        argparse.Namespace; updated global parameters.
        """
        self.input = parser
        self._version = 'STORM v.3.0'
        # assign the global/default parameters (to inner variables)
        self.numsims = kwargs.get('n_sims', NUMSIMS)
        self.numsimyrs = kwargs.get('n_sims_y', NUMSIMYRS)
        self.ptot_sc = kwargs.get('ptot_sc', PTOT_SC)
        self.ptot_sf = kwargs.get('ptot_sf', PTOT_SF)
        self.storm_sc = kwargs.get('storm_sc', STORMINESS_SC)
        self.storm_sf = kwargs.get('storm_sf', STORMINESS_SF)
        # self.parsing()

    def none_too(self, v):
        """
        assimilates NONE inputs from the command line.\n
        Parameters
        ----------
        v : char;
            none string.\n
        Returns
        -------
        updated input to NoneType or float.
        """
        # https://stackoverflow.com/a/48295546/5885810  (None in argparse)
        return None if v.lower() == 'none' else float(v)

    def update_arg(self, arguments):
        """
        redefines variables by their names, instead of relative to 'arguments'.\n
        Parameters
        ----------
        arguments : argparse.Namespace;
            global parameters to update.\n
        Returns
        -------
        argparse.Namespace; updated global parameters.
        """
        for x in list(vars(arguments).keys()):
            # # use this if you're not inside a (local) function
            # exec(f'if arguments.{x}:\n\t{x} = arguments.{x}')
            exec(f'if arguments.{x}:\n\tglobals()["{x}"] = arguments.{x}')
        # https://stackoverflow.com/a/2083375/5885810   (exec global... weird)
        return arguments

    # https://www.geeksforgeeks.org/command-line-arguments-in-python/
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    # https://stackoverflow.com/a/27616814/5885810  (parser case insensitive)
    def parsing(self,):
        """
        reads parameters from the command line.\n
        Parameters
        ----------
        None.\n
        Returns
        -------
        argparse.Namespace; updated global parameters.
        """
        # add the defaults
        self.input.add_argument(
            '-n', '--NUMSIMS', type=int, default=self.numsims,
            help='Number of simulations (default: %(default)s)'
            )
        self.input.add_argument(
            '-y', '--NUMSIMYRS', type=int, default=self.numsimyrs,
            help='Number of years per Simulation (default: %(default)s)'
            )
        self.input.add_argument(
            '-ps', '--PTOT_SC', default=self.ptot_sc, type=self.none_too,
            nargs='+',  # type=float,
            help='Relative change in the seasonal rain equally applied to '
            'every simulated year. (one signed scalar per Simulation).'
            )
        self.input.add_argument(
            '-pf', '--PTOT_SF', default=self.ptot_sf, type=self.none_too,
            nargs='+',
            help='Relative change in the seasonal rain progressively applied '
            'to every simulated year. (one signed scalar per Simulation).'
            )
        self.input.add_argument(
            '-ss', '--STORMINESS_SC', default=self.storm_sc, type=self.none_too,
            nargs='+', help='Relative change in the observed intensity equally '
            'applied to every simulated year. (one signed scalar per Simulation).'
            )
        self.input.add_argument(
            '-sf', '--STORMINESS_SF', default=self.storm_sf, type=self.none_too,
            nargs='+', help='Relative change in the observed intensity progressively '
            'applied to every simulated year. (one signed scalar per Simulation).'
            )
        self.input.add_argument(
            '--version', action='version', version=self._version
            )
        # Read arguments from command line
        args = self.input.parse_args()
        # print(args)
        return self.update_arg(args)


# %% file-naming

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
        n_sims : int; indicates the number of simulations/files to output.
        n_sims_y : int; indicates the number of years (per simulation) to run.
        out_path : char; where to store the simulations.\n
        Output -> list; containing output file-paths/names.
        """
        # assign the global/default parameters (to inner variables)
        self.numsims = kwargs.get('n_sims', NUMSIMS)
        self.numsimyrs = kwargs.get('n_sims_y', NUMSIMYRS)
        self.ptot_sc = kwargs.get('ptot_sc', PTOT_SC)
        self.ptot_sf = kwargs.get('ptot_sf', PTOT_SF)
        self.storm_sc = kwargs.get('storm_sc', STORMINESS_SC)
        self.storm_sf = kwargs.get('storm_sf', STORMINESS_SF)
        self.out_path = kwargs.get('out_path', OUT_PATH)
        # table relating core variables (Var2) to understandable meanings (Var3)
        self.wet_hash = DataFrame({
            'Var2':['PTOT', 'STORMINESS'],
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
        self.replicate()
        self.ncs = self.output_path()

    def replicate(self,):
        """
        replicates scalars for NUMSIMS (if only one was passed).\n
        Input: none.\n
        Output -> none; updated instances/globals of scaling parameters.
        """
        scalar = {
            'PTOT_SC': self.ptot_sc, 'PTOT_SF': self.ptot_sf,
            'STORMINESS_SC': self.storm_sc, 'STORMINESS_SF': self.storm_sf,
            }
        n_scal = np.array([len(x) for x in list(scalar.values())])
        if np.unique(n_scal).min() < self.numsims:
            warn(f'\nIncompatible Sizes in {list(scalar.keys())} and '
                 f'NUMSIMS == {self.numsims}.\nSTORM will only use the values '
                 'of the first Simulation, so they can be passed to all '
                 f'{self.numsims} Simulations.')
            for x in list(scalar.keys()):
                exec(f'globals()["{x}"] = np.repeat(scalar[x], self.numsims)')
        # print(PTOT_SC, PTOT_SF)
        self.ptot_sc = PTOT_SC
        self.ptot_sf = PTOT_SF
        self.storm_sc = STORMINESS_SC
        self.storm_sf = STORMINESS_SF

    def output_path(self,):
        """
        generates the names of the output nc-files.\n
        Input: none.\n
        Output -> list; containing output file-paths/names.
        """
        # infer scenarios
        PTOT_scene = self.infer_scenario(
            self.ptot_sc, self.ptot_sf, self.tab_ptot, self.tab_sign
            )
        STORMINESS_scene = self.infer_scenario(
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
            f'{c.strip()}.nc', range(self.numsims), PTOT_scene, STORMINESS_scene))

        # print the CORE INFO
        print('\nRUN SETTINGS')
        print('************\n')
        print(f'number of simulations: {self.numsims}')
        print(f'years per simulation : {self.numsimyrs}')
        for j in self.wet_hash.Var2:
            print(f'{self.wet_hash[self.wet_hash.Var2.isin([j])].Var3.iloc[0]} scenarios '
                  f'({" | ".join([f"sim{x+1}" for x in range(self.numsims)])}):  '
                  f'{ " | ".join(map(eval, [f"{j}_scene[{x}].center(8," ")" for x in range(self.numsims)]))}')
            # 8 because 'stormsT+' is the maximum length of these strings
        # https://stackoverflow.com/a/25559140/5885810  (string no sign)
        # https://www.delftstack.com/howto/python/python-pad-string-with-spaces/
        # https://stackoverflow.com/a/45120812/5885810  (print/create string padding)
        print('\nOutput paths:')
        print(*[(k.ljust(max(map(len, nc_paths)), ' ')).rjust(
            max(map(len, nc_paths)) + 4, ' ') for k in nc_paths], sep='\n',)
        print('')
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

# %% assert

def assertion(wet_hash):
    """
    performs some assertions to test the validity of input parameters.\n
    Input ->
    *wet_hash* : pandas.DataFrame; var-names (Var2) related to outputs (Var3).\n
    Output -> none; assertions.
        """
    for j in wet_hash.Var2:
        SC = f'{j}_SC'
        SF = f'{j}_SF'
        # print(NUMSIMS)
        # does each dimension/season have the same length among them??
        if ~np.isin(np.unique([len(eval(x)) for x in [SC, SF]]), 1).all():
            warn(f'\nIncompatible Sizes in {SC} and {SF}!\nSTORM will '
                 'only use the values of the first Simulation, so they '
                 f'can be passed to all {NUMSIMS} Simulations.')
        # is each dimension/simulation as long as 1 or NUMSIMYRS??
        for i in range(NUMSIMS):
            # does the progression factor (SF) reduces below 0 rain?
            assert 1 + eval(f'{SF}[{i}]') * (NUMSIMYRS - 1) >= 0,\
                'Scalar Overflooding!\nPlease, ensure that over the '\
                f'{NUMSIMYRS}-years, the trend scalar {SF} in year '\
                f'{i} does not produce negaative rainfall.'
            # is the scalar factor (SC) below 0 rain?
            assert 1 + eval(f'{SC}[{i}]') >= 0, 'Scalar Overflooding!\n'\
                f'Please, ensure that step change scalar {SC} in year {i} '\
                'does not produce negaative rainfall.'
        some_sum = np.nansum((
            np.asarray(eval(SC), dtype='f8') / np.asarray(eval(SC), dtype='f8'),
            2 * np.asarray(eval(SF), dtype='f8') / np.asarray(eval(SF), dtype='f8')
            ), axis=0, dtype=np.int32)
        # the integer 3 distinguishes between 'stepchange' & 'scaling_factor'
        assert 3 not in some_sum, f'{j}_SCENARIO not valid!\n'\
            f'Please, ensure that either {SC} or {SF} (or both!) '\
            'are set to 0 (zero), for any given season of the run.'
    # check that the DEM exists (if you're using Z_CUTS)
    if Z_CUTS:
        assertdem = f'NO DEM_FILE!\n'\
            'You chose to model rainfall via copula.setting, i.e., TACTIC == 2'\
            f' and Z_CUTS != []. Nevertheless, the path to the DEM {DEM_FILE} '\
            'was not correctly set up or the file does not exist.\n'\
            'Please ensure that the DEM file exists in the correct path.'
        # https://stackoverflow.com/a/82852/5885810  (files exists?)
        # https://stackoverflow.com/a/40183030/5885810  (assertion error)
        # https://stackoverflow.com/a/6095782/5885810  (multiple excepts)
        # https://stackoverflow.com/a/46827349/5885810  (None-Type error)
        # try:
        #     Path(abspath(join(parent_d, DEM_FILE))).resolve(strict=True)
        # except (FileNotFoundError, TypeError):
        #     raise AssertionError(assertdem)
        # https://stackoverflow.com/a/60324874/5885810  (using os.path.exists)
        # https://therenegadecoder.com/code/how-to-check-if-a-file-exists-in-python/
        if DEM_FILE is None or not exists(abspath(join(parent_d, DEM_FILE))):
            raise AssertionError(assertdem)
    # check that the SHP exists (always!)
    assertshp = f'NO SHP_FILE!\n'\
        f'The path to the SHP file {SHP_FILE} was not correctly '\
        'set up or the file does not exist.\nPlease ensure that the SHP '\
        'file exists in the correct path.'
    if SHP_FILE is None or not exists(abspath(join(parent_d, SHP_FILE))):
        raise AssertionError(assertshp)



# %% run

if __name__ == '__main__':

    parser = ArgumentParser(description='STOchastic Rainstorm Model [STORM v3.0]')
    # call class
    update = parse(parser)
    # run the parsing
    update.parsing()

    willkommen = welcome()
    # willkommen.ncs
    assertion(willkommen.wet_hash)
