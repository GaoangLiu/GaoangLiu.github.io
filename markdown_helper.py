import re
import os
import sys


def math_format(text):
    '''Replace math formula environment signals MSB/msb by $$ .
    '''
    r = re.sub(r'MSB', '$$', text, flags=re.IGNORECASE)
    return r 


if __name__ == "__main__":
    mdfile = sys.argv[1]

    '''read, modify and write back'''
    text = open(mdfile, 'r').read()
    text_new = math_format(text)
    with open(mdfile, 'w') as f:
        f.write(text_new)
