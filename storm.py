"""
To run this script type:
    "python storm.py"    (from your CONDA environment or Terminal)
    "%%python storm.py"  (from your Python console)
"""


# %% functions

def read():
    """
    parse and assertion of some input parameters.\n
    Parameters
    ----------
    None.\n
    Returns
    -------
    argparse.Namespace : updated parameters/globals.
    """
    from argparse import ArgumentParser
    from checks_ import parse
    parser = ArgumentParser(
        description='STOchastic Rainstorm Model [STORM v3.0]'
        )
    # call class
    arg_upd = parse(parser)
    # run the parsing
    arg_upd = arg_upd.parsing()
    return arg_upd


def test(arg_upd):
    """
    parse and assertion of some input parameters.\n
    Parameters
    ----------
    arg_upd : argparse.Namespace; with parameters/globals to update.\n
    Returns
    -------
    list : list of simulation output-paths (chars).
    """
    from checks_ import welcome, assertion
    # print(arg_upd)
    willkommen = welcome()
    assertion(willkommen.wet_hash)
    return willkommen.ncs


def compute_storm(one, two):
    """
    calls scripts in 'rainfall.py' so STORMS can be computed.\n
    Parameters
    ----------
    one : argparse.Namespace; parameters read from the command prompt.
    two : list; simulation output-paths.\n
    Returns
    -------
    None : produces nc.files.
    """
    # import here these heavy modules so there's little toll when calling for
    # ... --help or --version in 'ArgumentParser'.
    from rainfall import update_par, replicate_, wrapper
    update_par(one)
    replicate_()
    wrapper(two, one.SEED_YEAR)


# %% run

if __name__ == '__main__':

    up_args = read()
    nc_name = test(up_args)
    compute_storm(up_args, nc_name)
