#!/usr/bin/env python
import codefast as cf
from codefast.axe import Axe
from dofast.pipe import author
import ast

if __name__ == "__main__":
    cnt = sum(
        len(cf.io.reads(f).split(' ')) for f in cf.io.walk('_drafts/2021/'))
    bs = ast.literal_eval(author.get('blog_summary'))
    latest_date = sorted(list(bs))[-1]
    cf.info('latest date {}'.format(latest_date))
    print("Today's process {}. Total {}".format(cnt - bs[latest_date], cnt))
    bs[Axe().today] = cnt
    author.set('blog_summary', bs)
