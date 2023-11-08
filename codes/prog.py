import argparse
import textwrap
parser = argparse.ArgumentParser(
     prog='PROG',
     formatter_class=argparse.RawDescriptionHelpFormatter,
     description=textwrap.dedent('''\
         Please do not mess up this text!
         --------------------------------
             1. information line 1
             2. information line 2
             ...
             998. information line 998
         '''))
# parser = argparse.ArgumentParser(prog="simple_script", epilog="Oh, man, this is awesome.")
parser.add_argument("-d", "--delete", help="remove invalid account from")
args = parser.parse_args()

