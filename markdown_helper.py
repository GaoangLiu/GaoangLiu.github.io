import re
import os
import sys


def math_format(text):
    '''Replace math formula environment signals MSB/msb by $$ .
    '''
    # ps = set(('MSB', '$$'), ('vtline', "\|"),
    # ('mynotice', '<img src="https://git.io/JkKbw" width="50px" alt="Notice">'))
    # for a, b in ps:
    r = re.sub(r'MSB', '$$', text, flags=re.IGNORECASE)
    r = re.sub(r"vtline", "\|", r, flags=re.IGNORECASE)
    r = re.sub(r'mynotice', '<img src="https://git.io/JkKbw" width="50px" alt="Notice">', r, flags=re.IGNORECASE)
    ps = "yccn:隐层/sjy:神经元/jhhs:激活函数/lccn:线性组合/inputcn:输入/outputcn:输出/wlcn:网络"
    for p in ps.split('/'):
        old, new = p.split(':')
        r = r.replace(old, new)
    return r 


if __name__ == "__main__":
    mdfile = sys.argv[1]

    '''read, modify and write back'''
    text = open(mdfile, 'r').read()
    text_new = math_format(text)
    with open(mdfile, 'w') as f:
        f.write(text_new)
