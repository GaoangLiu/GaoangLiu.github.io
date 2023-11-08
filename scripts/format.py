import re
import sys
import codefast as cf

for file in cf.io.walk('_posts/2022/'):
    if file.endswith('.md'):
        rl, rold= '',''
        with open(file, 'r') as f:
            rl = f.read()
            rold = rl
            rl = re.sub(r'\$(.*?)\$', r'$$\1$$', rl)
            rl = re.sub(r'\$\$\$\$', r'$$', rl)

        if rl == '':
            print("Something went wrong")
            continue

        with open(file, 'w') as wf:
            wf.write(rl)



def format_header(f):
    ss = ""
    with open(f, 'r') as fh:
        for line in fh.readlines():
            if line.startswith("### "):
                line = line.replace("### ", '')
                line = '<h3 style="font-size:bold; font-family: STKaiti, KaiTi">☘️☘️☘️ {} </h3> \n'.format(
                    line)
            elif line.startswith("## "):
                line = line.replace("## ", '')
                line = '<h2 style="font-size:bold; font-family: STHeiti, SimHei, STKaiti, KaiTi">🌲🌲 {} </h2> \n'.format(
                    line)
            elif line.startswith("# "):
                line = line.replace("# ", '')
                line = '<h1 style="font-size:bold;">🌵{} </h1> \n'.format(line)
            ss += line

    newfile = f.split('.')[0] + '_archived.md'
    with open(newfile, 'w') as fh:
        fh.write(ss)



