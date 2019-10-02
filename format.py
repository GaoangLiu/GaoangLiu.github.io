import os 
import re 
import json 
import sys

if len(sys.argv) < 2: 
    print("A file must be specified.")
    exit(0)

file = sys.argv[1]
rl = ''
with open(file, 'r') as f:
    rl = f.read()
    rl = re.sub(r'\\dd\[(.*?)\]', r' $$\1$$ ', rl)

if rl == '':
    print("Something went wrong")
    exit(0)
    
with open(file, 'w') as wf:
    wf.write(rl)

