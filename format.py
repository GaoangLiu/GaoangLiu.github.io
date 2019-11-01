import os 
import re 
import json 
import sys

# if len(sys.argv) < 2: 
#     print("A file must be specified.")
#     exit(0)

# file = sys.argv[1]
# rl = ''
# with open(file, 'r') as f:
#     rl = f.read()
#     rl = re.sub(r'\\dd\[(.*?)\]', r' $$\1$$ ', rl)

# if rl == '':
#     print("Something went wrong")
#     exit(0)
    
# with open(file, 'w') as wf:
#     wf.write(rl)

def format_header(f):
    ss = ""
    with open(f, 'r') as fh:
        for line in fh.readlines():
            if line.startswith("### "):
                line = line.replace("### " , '')
                line = '<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ {} </h3> \n'.format(line)
            elif line.startswith("## "):
                line = line.replace("## " , '')
                line = '<h2 style="font-size:bold; font-family: STHeiti, SimHei, STKaiti, KaiTi">🌲🌲 {} </h2> \n'.format(line)
            elif line.startswith("# "):
                line = line.replace("# " , '')
                line = '<h1 style="font-size:bold;">🌵{} </h1> \n'.format(line)
            ss += line
    
    newfile = f.split('.')[0] + '_archived.md'
    with open(newfile, 'w') as fh:
        fh.write(ss)

format_header('_drafts/ml-notes.md')

