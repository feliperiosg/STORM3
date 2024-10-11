"""
To run this script type:
    "python storm.py"    (from your CONDA environment or Terminal)
    "%%python storm.py"  (from your Python console)
"""


#~ executes the PARSER (first), so no need to upload all.libs to ask.4.help ~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def READ():
    import argparse
# # maybe just..
#     from parse_input import PARCE
    import parse_check
    global argsup
    parser = argparse.ArgumentParser(description='STOchastic Rainstorm Model [STORM v3.0]')
# updated args
    argsup = parse_check.PARCE(parser)
    # print( argsup )


#~ checks the validity of the SOFT-CORE PARAMETERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def TEST():
    from parse_check import ASSERT, welcome, PAR_UPDATE
    global NC_NAMES
    PAR_UPDATE(argsup)
    willkommen = welcome()
    NC_NAMES = willkommen.ncs
    ASSERT(willkommen.wet_hash)


#~ all the heavy work is done in 'rainfall.py' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def RAIN():
    from rainfall import wrapper, PAR_UPDATE
    PAR_UPDATE(argsup)
    wrapper(NC_NAMES)


if __name__ == '__main__':
    READ()
    TEST()
    RAIN()