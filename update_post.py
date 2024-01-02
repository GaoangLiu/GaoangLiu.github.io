#!/usr/bin/env python
import arrow
import os
import re

import codefast as cf


class Blog(object):
    def _export_to_post(self, fpath):
        lo = self._get_layouts(fpath)
        if lo['layout'] == 'draft': # keep it to local
            return 
        lo['layout'] = 'post'
        nm = lo['date'] + '-' + lo['title'].replace(' ', '-') + '.md'
        year = lo['date'].split('-')[0]
        postpath = f'_posts/{year}/{nm}'
        if not cf.io.exists(f'_posts/{year}'):
            os.makedirs(f'_posts/{year}')

        head = "---\n"
        for k, v in lo.items():
            head += k + ": "
            # Some theme divide tags by ','
            head += " ".join(map(lambda e: e.lower().replace(' ', '_'),
                                 v)) if type(v) == list else v
            head += "\n"
        head += "author: berrysleaf\n---"
        # print(postpath, head)
        con = self._get_contents(fpath)
        con = re.sub(r'\$(.*?)\$', r'$$\1$$', con)
        con = re.sub(r'\$\$\$\$', r'$$', con)
        open(postpath, 'w').write(head + con)

    def _get_contents(self, fpath):
        f = open(fpath, 'r').read()
        cs = re.findall(r'---.*?---(.*)', f, re.DOTALL)[0]

        if "{:toc}" not in cs:
            i = 0
            while i < min(len(cs), 30) or (i<len(cs) and cs[i] != '\n'):
                i += 1
            cs = cs[:i] + '\n\n\n' + cs[i:]
            return "\n* content\n{:toc}\n" + cs
        return cs

    def _get_layouts(self, fpath):
        # Get layout info (title, tag, categorical, etc)
        layouts = dict()
        with open(fpath, 'r') as f:
            header = re.findall(r'---(.*?)---', f.read(), re.DOTALL)[0]
            # print(header)
            for line in header.split('\n'):
                if ':' not in line:
                    continue
                key, value = line.split(':')
                value = re.sub('[\[\]]', '', value).strip()
                layouts.update((k, value) for k in ('layout', 'title', 'date')
                               if k in key)

                # Parse out tags and categories
                if 'tags' in key:
                    layouts['tags'] = [t.strip() for t in value.split(',')]

            cats = re.findall(r'cat[eo]g[oe]ries:(.*)', header, re.DOTALL)
            if cats:
                cats = list(
                    filter(
                        lambda e: e,
                        map(lambda e: e.strip('-').strip(),
                            cats[0].split('\n'))))
            layouts['categories'] = cats
        # print(layouts)
        return layouts

    def post_drafts(self):
        # os.remove('_posts/*')
        draft_dir = '_drafts'
        for root, subdirs, files in os.walk(draft_dir):
            if len(subdirs) == 0:
                for f in files:
                    full_path = f'{root}/{f}'
                    print(full_path)
                    if os.path.isfile(full_path) and full_path.endswith('.md'):
                        self._export_to_post(full_path)


class ProgressTracker(object):
    def __init__(self) -> None:
        self._logs = cf.l(cf.js('assets/progress.json')) if cf.io.exists('assets/progress.json') else cf.l([('1970-01-01', 0)])

    def count_cn(self, md_file: str) -> int:
        """Count Chinese Character"""
        with open(md_file, 'r') as f:
            return len(re.findall(r'[\u4e00-\u9fa5]', f.read()))

    def run(self):
        sm = sum(cf.l(cf.io.walk('_drafts', depth=6)).map(self.count_cn).data)
        today = arrow.now().format('YYYY-MM-DD')
        self._logs.sort()
        if self._logs[-1][0] == today:
            presum = sum([v for _, v in self._logs.data[:-1]])
            self._logs[-1] = (today, sm - presum)
        else:
            self._logs.print()
            presum = sum([v for _, v in self._logs.data])
            self._logs.append([today, sm-presum])
        self._logs.print()
        cf.js.write(self._logs.data, 'assets/progress.json')


def clean_previous_posts():
    os.system('rm _posts/2023/*')
    os.system('rm _posts/2024/*')

if __name__ == "__main__":
    clean_previous_posts()
    Blog().post_drafts()
    ProgressTracker().run()
