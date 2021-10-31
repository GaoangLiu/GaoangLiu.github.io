#!/usr/bin/env python
import ast

import codefast as cf
from codefast.axe import Axe
from dofast.pipe import author

# js = {
#     '2021-10-09': 5293,
#     '2021-10-10': 5862,
#     '2021-10-12': 6020,
#     '2021-10-29': 8348,
#     '2021-10-30': 8348
# }

if __name__ == "__main__":
    cnt = sum(
        len(cf.io.reads(f).split(' ')) for f in cf.io.walk('_drafts/2021/'))
    log_dict = ast.literal_eval(author.get('blog_summary'))
    today = Axe.today()
    log_dict[today] = cnt
    bs = [(d, c) for d, c in log_dict.items()]
    bs.sort()
    dummy = [(None, 0)] + bs
    log_dict[Axe().today()] = cnt

    for pre, cur in zip(dummy, bs):
        _diff = cur[1] - pre[1]
        text_formatted = cf.fp.red(
            " +{}".format(_diff)) if _diff > 0 else cf.fp.cyan(
                " -{}".format(_diff))
        print(cur[0], cur[1], text_formatted)
    # print(bs)
    author.set('blog_summary', log_dict)

